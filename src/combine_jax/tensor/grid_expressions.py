"""Import coordinate-bound expressions into batched, branch-safe tensor operations."""
from dataclasses import dataclass

import numpy as np

from ..graph import UnsupportedModelError
from .scalar import SCALAR_CLASSES,OPERATIONS,UNARY,BINARY,operation as scalar_operation


GRID_OPERATIONS=frozenset('gridexpr_'+op for op in OPERATIONS|{'select'})


def grid_operation(op,values,b,xp=np):
    table=values[b['children']];active=table[:,0,:]!=0
    args=xp.where(active[:,:,None],xp.swapaxes(table[:,1:,:],1,2),1.)
    kind=op.removeprefix('gridexpr_')
    if kind=='select':result=xp.where(args[...,0]!=0,args[...,1],args[...,2])
    else:
        if xp is np:
            from scipy import special
        else:
            from jax.scipy import special
        result=scalar_operation(kind,args,b['operand_mask'][:,None,:],xp,special)
    return xp.asarray(xp.where(active,result,0.),dtype=xp.float64)


def validate_grid_fields(op,b,rows):
    kind=op.removeprefix('gridexpr_')
    if op not in GRID_OPERATIONS:raise ValueError('Unknown coordinate expression operation')
    mask=b['operand_mask'];children=b['children']
    if mask.dtype!=bool or mask.shape!=(len(children),children.shape[1]-1):raise ValueError('Invalid coordinate expression mask')
    arity=1 if kind in UNARY else 2 if kind in BINARY else 3 if kind in ('select','asymmetric_log') else None
    if mask.shape[1]<1 or (arity is not None and (mask.shape[1]!=arity or not np.all(mask))):
        raise ValueError('Invalid coordinate expression arity')


@dataclass
class Value:
    fixed: object=None
    reference: object=None


class GridBuilder:
    """Only import-time node/AST walks; runtime kernels see typed batches."""
    def __init__(self,nodes,coordinates,rows,lookup,dynamic,constant,emit,fixed,label):
        self.nodes,self.coordinates,self.rows=nodes,coordinates,rows
        self.lookup,self.dynamic,self.constants=lookup,dynamic,constant
        self.emit,self.fixed,self.label=emit,fixed,label
        self.cache={};self.literal_cache={};self.certified=set();self.expression_cache={}

    def constant(self,value):
        if value not in self.literal_cache:
            self.literal_cache[value]=Value(fixed=np.full(self.rows,value,dtype=np.float64))
        return self.literal_cache[value]

    def reference(self,value):
        if value.reference is None:value.reference=self.fixed(value.fixed,self.label+':folded_expression')
        return value.reference

    def apply(self,op,arguments,active=None):
        active=self.constant(1.) if active is None else active
        if not arguments and op in ('sum','product'):return self.constant(0. if op=='sum' else 1.)
        if op not in OPERATIONS|{'select'}:raise UnsupportedModelError(f'{self.label}: unsupported coordinate opcode {op}')
        children=[active,*arguments];key=(op,*(id(v) for v in children))
        if key in self.expression_cache:return self.expression_cache[key]
        if all(v.fixed is not None for v in children):
            table=np.stack([v.fixed for v in children]);fields=dict(children=np.arange(table.size,dtype=np.int32).reshape(1,*table.shape),
                operand_mask=np.ones((1,len(arguments)),dtype=bool))
            with np.errstate(all='ignore'):result=grid_operation('gridexpr_'+op,table.reshape(-1),fields)[0]
            if not np.all(np.isfinite(result)):raise UnsupportedModelError(f'{self.label}: invalid active fixed coordinate expression {op}')
            value=Value(fixed=result)
        else:
            fields=dict(operand_mask=np.ones(len(arguments),dtype=bool))
            value=Value(reference=self.emit('gridexpr_'+op,self.rows,fields,[self.reference(v) for v in children],label=self.label))
        self.expression_cache[key]=value;return value

    def formula(self,tree,arguments,active=None):
        active=self.constant(1.) if active is None else active
        if active.fixed is not None and not np.any(active.fixed):return self.constant(0.)
        kind=tree[0]
        if kind=='constant':return self.constant(tree[1])
        if kind=='argument':return arguments[tree[1]]
        if kind in ('binary','compare'):
            binary={'+':'add','-':'sub','*':'mul','/':'div','**':'pow','<':'lt','<=':'le','>':'gt','>=':'ge','==':'eq','!=':'ne'}
            return self.apply(binary[tree[1]],[self.formula(t,arguments,active) for t in tree[2:]],active)
        if kind=='unary':
            value=self.formula(tree[2],arguments,active)
            return value if tree[1]=='+' else self.apply({'-':'neg','not':'not'}[tree[1]],[value],active)
        if kind=='call':return self.apply({'fabs':'abs','pow':'pow'}.get(tree[1],tree[1]),[self.formula(t,arguments,active) for t in tree[2]],active)
        if kind=='boolean':
            result=self.formula(tree[2][0],arguments,active)
            for term in tree[2][1:]:result=self.apply(tree[1],[result,self.formula(term,arguments,active)],active)
            return result
        if kind=='conditional':
            condition=self.formula(tree[1],arguments,active)
            yes=self.apply('and',[active,condition]);no=self.apply('and',[active,self.apply('not',[condition])])
            return self.apply('select',[condition,self.formula(tree[2],arguments,yes),self.formula(tree[3],arguments,no)],active)
        raise UnsupportedModelError(f'{self.label}: unsupported coordinate formula AST {kind}')

    def certify(self,name):
        if name is None or name in self.certified or name in self.lookup:return
        cls,children,_=self.nodes[name]
        if cls not in SCALAR_CLASSES:raise UnsupportedModelError(f'{name}: unsupported coordinate expression class {cls}')
        for n in children:self.certify(n)
        self.certified.add(name)

    def node(self,name):
        if name is None:return self.constant(0.)
        if name in self.cache:return self.cache[name]
        self.certify(name)
        if name in self.lookup:
            if name in self.dynamic:
                fields=dict(coefs=np.array([self.lookup[name]],dtype=np.int32),coef_mask=np.ones(1,dtype=bool),broadcast=np.zeros(self.rows))
                result=Value(reference=self.emit('coef_sum',self.rows,fields,coef_names=(name,),label=self.label+':'+name))
            else:result=self.constant(self.constants[name])
        elif name in self.coordinates:result=Value(fixed=np.asarray(self.coordinates[name],dtype=np.float64))
        else:
            cls,children,a=self.nodes[name];values=[self.node(n) for n in children]
            if cls=='RooFormulaVar':result=self.formula(a['ast'],values)
            elif cls in ('RooProduct','RooAddition'):result=self.apply('product' if cls=='RooProduct' else 'sum',values)
            elif cls=='RooRecursiveFraction':result=self.apply('product',[values[0],*[self.apply('sub',[self.constant(1.),v]) for v in values[1:]]])
            elif cls=='AsymPow':
                low,high,theta=values
                result=self.apply('exp',[self.apply('asymmetric_log',[theta,self.apply('log',[low]),self.apply('log',[high])])])
            elif cls=='ProcessNormalization':
                exponents=[self.apply('mul',[self.node(n),self.constant(v)]) for n,v in zip(a['symmetric'],a['log_kappa'],strict=True)]
                exponents += [self.apply('asymmetric_log',[self.node(n),self.constant(lo),self.constant(hi)]) for n,(lo,hi) in zip(a['asymmetric'],a['log_asymmetric'],strict=True)]
                result=self.apply('product',[self.constant(a['nominal']),self.apply('exp',[self.apply('sum',exponents)]),*[self.node(n) for n in a['other']]])
            else:raise UnsupportedModelError(f'{name}: missing scalar/coordinate binding or unsupported {cls}; profile dependencies cannot fold')
        self.cache[name]=result;return result
