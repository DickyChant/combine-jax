"""Synthetic observed-style datasets, shared by native v10 and JAX tests.

No analysis inputs. Data are deliberately different from model expectations.
"""
from array import array


def workspace(root, kind):
    if kind not in ('unweighted_unbinned','weighted_binned','empty_bins','mixed'):
        raise ValueError(kind)
    w=root.RooWorkspace('w')
    w.factory('x[1,0,4]');w.factory('mu[1,0.2,3]');w.factory('theta[0,-3,3]')
    w.var('x').setBinning(root.RooBinning(3,array('d',[0.,1.,2.,4.])))
    w.factory('expr::signal_rate("7*@0",mu)')
    w.factory('expr::background_rate("5*exp(0.1*@0)",theta)')
    w.factory('expr::slope("-0.3+0.05*@0",theta)')
    w.factory('Gaussian::theta_constraint(theta,theta_In[0],sigma[1])')
    w.var('theta_In').setConstant(True);w.var('sigma').setConstant(True)
    channels=['unbinned','binned'] if kind=='mixed' else ['unbinned' if kind=='unweighted_unbinned' else 'binned']
    keep=[]
    category=root.RooCategory('CMS_channel','')
    for label in channels:
        category.defineType(label)
        if label=='unbinned':
            w.factory('Exponential::signal_unbinned(x,slope)')
            w.factory('Uniform::background_unbinned(x)')
        else:
            for process,contents in [('signal',[3.,5.,2.]),('background',[5.,2.,4.])]:
                hist=root.TH1D(process+'_input','',3,array('d',[0.,1.,2.,4.]));hist.SetDirectory(0)
                for i,value in enumerate(contents,1):hist.SetBinContent(i,value)
                dh=root.RooDataHist(process+'_template','',root.RooArgList(w.var('x')),hist)
                pdf=root.RooHistPdf(process+'_binned','',root.RooArgSet(w.var('x')),dh)
                getattr(w,'import')(pdf,root.RooFit.RecycleConflictNodes())
                keep.extend([hist,dh,pdf])
        w.factory('SUM::sum_'+label+'(signal_rate*signal_'+label+',background_rate*background_'+label+')')
        w.factory('PROD::pdf_'+label+'(sum_'+label+',theta_constraint)')
    sim=root.RooSimultaneous('sim','',category)
    for label in channels:sim.addPdf(w.pdf('pdf_'+label),label)
    model=root.RooSimultaneousOpt(sim);model.SetName('model_s')
    getattr(w,'import')(model,root.RooFit.RecycleConflictNodes())
    obs=root.RooArgSet(w.var('x'),w.cat('CMS_channel'))
    mc=root.RooStats.ModelConfig('ModelConfig',w);mc.SetPdf(w.pdf('model_s'));mc.SetObservables(obs)
    mc.SetParametersOfInterest(root.RooArgSet(w.var('mu')))
    mc.SetNuisanceParameters(root.RooArgSet(w.var('theta')))
    mc.SetGlobalObservables(root.RooArgSet(w.var('theta_In')));getattr(w,'import')(mc)
    weighted=kind!='unweighted_unbinned'
    weight=root.RooRealVar('event_weight','',1.,0.,1e6)
    columns=root.RooArgSet(obs)
    if weighted:columns.add(weight)
    data=root.RooDataSet('data_obs','',columns,*([root.RooFit.WeightVar(weight)] if weighted else []))
    expected=[]
    for label in channels:
        w.cat('CMS_channel').setLabel(label)
        rows=[(.07,1.),(.42,1.),(.9,1.),(1.21,1.),(2.14,1.),(2.98,1.),(3.8,1.)] if label=='unbinned' else [(.5,3.5),(1.5,0. if kind in ('empty_bins','mixed') else 2.25),(3.,7.25)]
        for x,n in rows:
            w.var('x').setVal(x);data.add(columns,n);expected.append(dict(channel=label,x=x,weight=n))
    if kind in ('weighted_binned','empty_bins'):data=data.binnedClone('data_obs','')
    getattr(w,'import')(data)
    w.var('mu').setVal(1.);w.var('theta').setVal(0.)
    return w,dict(kind=kind,rows=expected,entries=int(data.numEntries()),sum_weights=float(data.sumEntries()),
                  data_class=str(data.ClassName()),is_weighted=bool(data.isWeighted()))
