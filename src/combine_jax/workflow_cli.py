"""Path-independent import, device fit and profile-scan commands."""
import os
from pathlib import Path


def add_commands(commands):
    from .runtime.options import add_host_memory_option
    from .runtime.cpu_threads import add_threads_option
    p=commands.add_parser('import',help='Import a selected workspace dataset into a neutral model directory')
    p.add_argument('workspace',type=Path);p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--data',required=True);p.add_argument('--analysis',choices=('hig23006','hhvv','blind','synthetic'),required=True)
    p.add_argument('--dataset-kind',choices=('asimov','observed'))
    p.add_argument('--preset',choices=('hig23006-kl',))
    p.add_argument('--workspace-name',default='w');p.add_argument('--snapshot');p.add_argument('--poi')
    p.add_argument('--set',default='');p.add_argument('--freeze',default='');p.add_argument('--ranges',default='')
    p.add_argument('--freeze-attributes',default='');p.add_argument('--mass',type=float)
    p.add_argument('--X-rtd',action='append',default=[])
    p.add_argument('--discrete',choices=('fixed','profile'),default='fixed')
    p.add_argument('--discrete-profile-mode',choices=('independent','native-batched'),default='independent',
        help='Native-batched profiles shared shapes in registered native category order; method validation remains separate')
    p.add_argument('--template-convention',choices=('native','v10_density'))
    p.add_argument('--cbshape-convention',choices=('native','root630'))
    p.add_argument('--profile-bounds-policy',choices=('reject','clip'))
    p.add_argument('--no-compact-density',action='store_true',
        help='Keep unused density slots for numerical/memory comparisons; default removes them exactly')
    for command in ('fit','scan'):
        p=commands.add_parser(command,help='Fit or profile a previously imported neutral model')
        add_host_memory_option(p)
        add_threads_option(p)
        p.add_argument('model',type=Path);p.add_argument('-o','--output',type=Path,required=True)
        p.add_argument('--device',choices=('cpu','gpu'),default='cpu')
        p.add_argument('--set',default='',help='Initial parameter assignments; use --freeze to fix them')
        p.add_argument('--freeze',default='');p.add_argument('--cache',type=Path)
        p.add_argument('--max-iterations',type=int,default=10000,
            help='Continuous iteration cap (default: 10000); EDM stopping threshold is unchanged')
        p.add_argument('--candidate-batch-size',type=int,default=4,
            help='Candidate width for a native-batched profile model')
        if command=='scan':
            p.add_argument('--poi',required=True)
            p.add_argument('--points',nargs='+',type=float,required=True)


def run(args):
    from .runtime.options import assignments,ranges
    os.environ['JAX_ENABLE_X64']='true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['JAX_PLATFORMS']='cuda' if getattr(args,'device','cpu')=='gpu' else 'cpu'
    if args.command=='import':
        from .workspace_import import import_workspace,hig_kl_options
        options=hig_kl_options() if args.preset else {}
        if args.preset and args.analysis!='hig23006':raise ValueError('HIG preset requires --analysis hig23006')
        for name in ('snapshot','poi','mass','template_convention','cbshape_convention','profile_bounds_policy'):
            if getattr(args,name) is not None:options[name]=getattr(args,name)
        options['parameters']={**options.get('parameters',{}),**assignments(getattr(args,'set'))}
        options['ranges']={**options.get('ranges',{}),**ranges(args.ranges)}
        for name in ('freeze','freeze_attributes'):
            if getattr(args,name):options[name]=getattr(args,name).split(',')
        flags=options.setdefault('runtime_defines',{})
        for text in args.X_rtd:
            name,_,value=text.partition('=');flags[name]=int(value or 1)
        _,result=import_workspace(args.workspace,args.output,data=args.data,analysis=args.analysis,
            dataset_kind=args.dataset_kind,workspace_name=args.workspace_name,discrete=args.discrete,
            discrete_profile_mode=args.discrete_profile_mode.replace('-','_'),
            compact_density=not args.no_compact_density,**options)
        print('Imported',result['channels'],'channels;',result['parameters_count'],'parameters;',args.output)
        return
    from .tensor.workflow import run_model
    report=run_model(args.model,args.output,device=args.device,initial=assignments(getattr(args,'set')),
        freeze=[n for n in args.freeze.split(',') if n],cache=args.cache,max_iterations=args.max_iterations,
        poi=getattr(args,'poi',None),points=getattr(args,'points',None),candidate_batch_size=args.candidate_batch_size,
        host_memory_limit_gib=args.host_memory_limit_gib)
    if report.get('global_recovery'):
        recovery=report['global_recovery']
        print('Global recovery from',recovery['seed_task']+';',
              'status',recovery['recovered_status'],'NLL improvement',
              format(recovery['nll_improvement'],'.8g')+';',
              'selected denominator',recovery['selected_baseline_task'])
    if report['status']!='complete_requires_native_validation':
        from .tensor.workflow import workflow_failures
        failures=workflow_failures(report)
        detail='; '.join(f['message'] for f in failures[:4])
        if len(failures)>4:detail+=f'; {len(failures)-4} more failures in the report'
        raise RuntimeError(detail+'. Evidence saved in '+str(args.output))
    print('Completed',len(report['fits']),'fits;',args.output)
