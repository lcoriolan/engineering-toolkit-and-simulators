// DESCRIPTION: Browser numerical work requires central lifecycle and step grants.
'use strict';
(() => {
  const endpoint = '/api/simulation/browser';
  async function request(payload) {
    const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),10000);
    try {
      const response = await fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload),signal:controller.signal});
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || 'Central simulation engine request failed.');
      return value;
    } finally { clearTimeout(timer); }
  }
  class Adapter {
    constructor(operation) { this.operation=operation; this.id=null; this.ready=false; this.busy=null; this.state='new'; this.lastResult=null; this.revision=0; }
    async work(grant, callback, result) {
      if (!grant.permit) { this.state=grant.state; return false; }
      const id=grant.id, token=grant.permit.token;
      try {
        callback(grant.permit.steps);
        const state=await request({action:'complete',id,token,result:result()});
        this.state=state.state; this.lastResult=state.result;
        if (state.state==='failed') throw new Error(state.error_type || 'Browser simulation failed.');
        return state;
      } catch (error) {
        this.ready=false; this.state='failed';
        try { await request({action:'fail',id,error_type:error.name || 'Error'}); } catch (_) { /* Original failure remains visible. */ }
        throw error;
      }
    }
    initialize(cells, callback, result) {
      const previous=this.busy, revision=++this.revision;
      this.ready=false;
      const run=(async()=>{
        if (previous) { try { await previous; } catch (_) {} }
        if(revision!==this.revision)return false;
        if (this.id) await request({action:'stop',id:this.id});
        const grant=await request({action:'create',operation:this.operation,cells});
        this.id=grant.id;
        const status=await this.work(grant,callback,result);
        this.ready=revision===this.revision && status && status.state==='running';
        return status;
      })();
      this.busy=run;
      run.finally(()=>{if(this.busy===run)this.busy=null;}).catch(()=>{});
      return run;
    }
    advance(steps, callback, result) {
      if (this.busy) return this.busy;
      if (!this.ready || !this.id) return Promise.reject(new Error('Central engine initialization is required before stepping.'));
      const run=(async()=>{
        const grant=await request({action:'grant',id:this.id,steps});
        return this.work(grant,callback,result);
      })();
      this.busy=run;
      run.catch(()=>{this.ready=false;this.state='failed';});
      run.finally(()=>{if(this.busy===run)this.busy=null;}).catch(()=>{});
      return run;
    }
    async reportFailure(error) {
      this.ready=false;this.state='failed';
      if(this.id) return request({action:'fail',id:this.id,error_type:error.name||'Error'});
    }
    async control(action) {
      if (!this.id) return null;
      const state=await request({action,id:this.id});
      this.state=state.state;
      if (state.state==='stopped' || state.state==='failed') this.ready=false;
      return state;
    }
  }
  function bind(ctor, operation, cells) {
    const reset=ctor.prototype.reset, step=ctor.prototype.step, dispose=ctor.prototype.dispose;
    ctor.prototype.reset=function(...args) {
      this.simulationEngine ||= new Adapter(operation);
      return this.simulationEngine.initialize(cells(this),()=>reset.apply(this,args),()=>this.diagnostics());
    };
    ctor.prototype.step=function(n=1) {
      if (!this.simulationEngine) return Promise.reject(new Error('Central engine initialization is required before stepping.'));
      return this.simulationEngine.advance(n,count=>step.call(this,count),()=>this.diagnostics());
    };
    ctor.prototype.setPaused=function(paused) { return this.simulationEngine?.control(paused?'pause':'resume') || Promise.resolve(null); };
    ctor.prototype.stopSimulation=function() { return this.simulationEngine?.control('stop') || Promise.resolve(null); };
    if(dispose) ctor.prototype.dispose=function(...args) {
      const pending=this.simulationEngine?.busy;
      if(this.simulationEngine)this.simulationEngine.ready=false;
      const finish=async()=>{if(pending){try{await pending;}catch(_){}}await this.stopSimulation();return dispose.apply(this,args);};
      return finish();
    };
  }
  globalThis.UnifiedSimulationBrowser = Object.freeze({API_VERSION:1, Adapter, bind});
})();
