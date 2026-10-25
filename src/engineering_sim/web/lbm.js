// DESCRIPTION: WebGL2 float-texture D2Q9 BGK solver, with link bounce-back obstacles.
'use strict';
class AirflowLBM{
  /** Allocate a 384×192 lattice. x is downstream, y upward, dt=dx=rho0=1.
   * Stored populations are post-collision. Each pass pulls them, reflects
   * missing solid links at their midpoint, and relaxes toward equilibrium.
   * This is a coarse, low-Mach laminar experiment, not a validated aero polar.
   */
  constructor(canvas,width=384,height=192){
    this.canvas=canvas;this.w=width;this.h=height;this.chord=width/6;this.u=.04;this.steps=0;
    const gl=this.gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
    if(!gl||!gl.getExtension('EXT_color_buffer_float'))throw Error('WebGL2 with float render targets is unavailable. No simulation is running.');
    const info=gl.getExtension('WEBGL_debug_renderer_info');this.renderer=info?gl.getParameter(info.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
    this.software=/swiftshader|llvmpipe|software/i.test(this.renderer);
    // A vertex-ID full-screen triangle avoids mesh buffers and shared-edge seams.
    const vertex=`#version 300 es
    void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}`;
    const common=`#version 300 es
    precision highp float;precision highp int;
    uniform sampler2D a;uniform sampler2D b;uniform sampler2D c;uniform sampler2D mask;
    uniform ivec2 size;uniform float inlet;uniform float tau;
    const ivec2 e[9]=ivec2[9](ivec2(0,0),ivec2(1,0),ivec2(0,1),ivec2(-1,0),ivec2(0,-1),ivec2(1,1),ivec2(-1,1),ivec2(-1,-1),ivec2(1,-1));
    const int opp[9]=int[9](0,3,4,1,2,7,8,5,6);
    const float wt[9]=float[9](4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.);
    float f(ivec2 p,int i){p=clamp(p,ivec2(0),size-1);if(i<4)return texelFetch(a,p,0)[i];if(i<8)return texelFetch(b,p,0)[i-4];return texelFetch(c,p,0).r;}
    bool solid(ivec2 p){return texelFetch(mask,clamp(p,ivec2(0),size-1),0).r>.5;}
    float eq(int i,float rho,vec2 u){float eu=dot(vec2(e[i]),u);return wt[i]*rho*(1.+3.*eu+4.5*eu*eu-1.5*dot(u,u));}
    vec3 macro(ivec2 p){float rho=0.;vec2 mom=vec2(0);for(int i=0;i<9;i++){float fi=f(p,i);rho+=fi;mom+=fi*vec2(e[i]);}return vec3(mom/max(rho,1e-9),rho);}
    `;
    this.compute=this.program(vertex,common+`
    layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;layout(location=2) out vec4 o2;
    void main(){ivec2 p=ivec2(gl_FragCoord.xy);float v[9];float rho=0.;vec2 mom=vec2(0);
      bool reservoir=p.x==0||p.y==0||p.y==size.y-1;bool body=solid(p);
      for(int i=0;i<9;i++){
        ivec2 src=p-e[i];
        // Right boundary copies the interior distribution (zero gradient).
        // Other external edges are fixed equilibrium reservoirs, not walls.
        v[i]=body?eq(i,1.,vec2(0)):(reservoir?eq(i,1.,vec2(inlet,0)):(p.x==size.x-1?f(p-ivec2(1,0),i):(solid(src)?f(p,opp[i]):f(src,i))));
        rho+=v[i];mom+=v[i]*vec2(e[i]);
      }
      vec2 u=mom/max(rho,1e-9);for(int i=0;i<9;i++)v[i]-=(v[i]-eq(i,rho,u))/tau;
      o0=vec4(v[0],v[1],v[2],v[3]);o1=vec4(v[4],v[5],v[6],v[7]);o2=vec4(v[8],0,0,0);
    }`);
    this.renderProgram=this.program(vertex,common+`
    uniform vec2 viewport;uniform int field;out vec4 color;
    void main(){ivec2 p=ivec2(gl_FragCoord.xy/viewport*vec2(size));
      if(solid(p)){color=vec4(.96,.72,.4,1);return;}
      vec3 m=macro(p);vec3 col;
      if(field==0){float t=clamp(length(m.xy)/inlet/1.8,0.,1.);
        col=t<.55?mix(vec3(.04,.08,.22),vec3(.08,.49,.67),t/.55):t<.8?mix(vec3(.08,.49,.67),vec3(.4,.87,.7),(t-.55)/.25):mix(vec3(.4,.87,.7),vec3(1,.67,.33),(t-.8)/.2);
      }else{float omega=(macro(p+ivec2(1,0)).y-macro(p-ivec2(1,0)).y-macro(p+ivec2(0,1)).x+macro(p-ivec2(0,1)).x)*.5;
        float t=clamp(omega/inlet*10.,-1.,1.);col=mix(vec3(.08,.14,.2),t>0.?vec3(1,.5,.2):vec3(.1,.6,1),abs(t));}
      color=vec4(col,1);
    }`);
    this.states=[this.state(),this.state()];this.front=0;
    this.maskTexture=this.texture(gl.R8,gl.RED,gl.UNSIGNED_BYTE,null);
    this.mask=new Uint8Array(width*height);
  }
  /** Compile/link with errors surfaced to the UI; never fake unsupported GPU output. */
  program(v,f){const gl=this.gl,p=gl.createProgram();for(const [type,src] of [[gl.VERTEX_SHADER,v],[gl.FRAGMENT_SHADER,f]]){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));gl.attachShader(p,s);gl.deleteShader(s);}gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
  /** Nearest filtering preserves individual lattice populations; no interpolation. */
  texture(internal,format,type,data){const gl=this.gl,t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,internal,this.w,this.h,0,format,type,data);return t;}
  /** Three MRT float textures hold f0..f8; unused components remain zero. */
  state(){const gl=this.gl,textures=Array.from({length:3},()=>this.texture(gl.RGBA32F,gl.RGBA,gl.FLOAT,null)),frame=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,frame);textures.forEach((t,i)=>gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.TEXTURE_2D,t,0));gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1,gl.COLOR_ATTACHMENT2]);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Float MRT framebuffer incomplete');return {textures,frame};}
  /** Rasterize normalized section at cell centers. Positive incidence raises the
   * leading edge: rotate downstream chord clockwise about its quarter chord.
   * CPU rasterization happens only on reset; all time integration is on GPU.
   */
  reset(section,re=80,incidence=4){
    if(!Number.isFinite(re)||re<40||re>160||!Number.isFinite(incidence)||Math.abs(incidence)>12)throw Error('Tunnel range is Re 40–160 and incidence ±12°');
    this.re=re;this.incidence=incidence;this.tau=.5+3*this.u*this.chord/re;this.steps=0;this.front=0;
    const a=-incidence*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
    const poly=section.map(([x,y])=>[this.w*.3+this.chord*((x-.25)*c-y*s),this.h/2+this.chord*((x-.25)*s+y*c)]);
    this.mask.fill(0);
    // Crossing-number polygon test includes the full modified section outline.
    for(let y=0;y<this.h;y++)for(let x=0;x<this.w;x++){let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const [xi,yi]=poly[i],[xj,yj]=poly[j];if((yi>y+.5)!==(yj>y+.5)&&(x+.5<(xj-xi)*(y+.5-yi)/(yj-yi)+xi))inside=!inside;}if(inside)this.mask[y*this.w+x]=255;}
    const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,this.w,this.h,gl.RED,gl.UNSIGNED_BYTE,this.mask);
    const e=[[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],w=[4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36];
    const arrays=Array.from({length:3},()=>new Float32Array(this.w*this.h*4));
    for(let n=0;n<this.mask.length;n++){const u=this.mask[n]?0:this.u;for(let i=0;i<9;i++){const eu=e[i][0]*u;arrays[Math.floor(i/4)][n*4+i%4]=w[i]*(1+3*eu+4.5*eu*eu-1.5*u*u);}}
    for(const state of this.states)state.textures.forEach((t,i)=>{gl.bindTexture(gl.TEXTURE_2D,t);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,this.w,this.h,gl.RGBA,gl.FLOAT,arrays[i]);});
  }
  /** Bind read-only current state; output always uses the other framebuffer. */
  bind(program){const gl=this.gl;gl.useProgram(program);[...this.states[this.front].textures,this.maskTexture].forEach((t,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,t);gl.uniform1i(gl.getUniformLocation(program,['a','b','c','mask'][i]),i);});gl.uniform2i(gl.getUniformLocation(program,'size'),this.w,this.h);gl.uniform1f(gl.getUniformLocation(program,'inlet'),this.u);gl.uniform1f(gl.getUniformLocation(program,'tau'),this.tau);}
  /** Advance an integer number of lattice timesteps, with no wall-clock scaling. */
  step(count=1){const gl=this.gl;gl.viewport(0,0,this.w,this.h);for(let i=0;i<count;i++){this.bind(this.compute);gl.bindFramebuffer(gl.FRAMEBUFFER,this.states[1-this.front].frame);gl.drawArrays(gl.TRIANGLES,0,3);this.front=1-this.front;this.steps++;}}
  /** Color the current numerical field; rendering never advances the solver. */
  draw(field=0){const gl=this.gl;this.bind(this.renderProgram);gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.uniform2f(gl.getUniformLocation(this.renderProgram,'viewport'),this.canvas.width,this.canvas.height);gl.uniform1i(gl.getUniformLocation(this.renderProgram,'field'),field);gl.drawArrays(gl.TRIANGLES,0,3);}
  /** Read populations for diagnostics/tests/export; this synchronizes the GPU. */
  read(){const gl=this.gl,state=this.states[this.front];gl.bindFramebuffer(gl.FRAMEBUFFER,state.frame);return state.textures.map((_,i)=>{const out=new Float32Array(this.w*this.h*4);gl.readBuffer(gl.COLOR_ATTACHMENT0+i);gl.readPixels(0,0,this.w,this.h,gl.RGBA,gl.FLOAT,out);return out;});}
  /** Monitor density/Mach drift and non-finite values, not aerodynamic accuracy. */
  diagnostics(){const arrays=this.read(),e=[[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]];let min=Infinity,max=-Infinity,maxU=0,error=0,count=0,finite=true;
    for(let n=0;n<this.mask.length;n++){if(this.mask[n])continue;let rho=0,mx=0,my=0;for(let i=0;i<9;i++){const f=arrays[Math.floor(i/4)][n*4+i%4];rho+=f;mx+=f*e[i][0];my+=f*e[i][1];}const ux=mx/rho,uy=my/rho;finite&&=Number.isFinite(rho)&&Number.isFinite(ux)&&Number.isFinite(uy);min=Math.min(min,rho);max=Math.max(max,rho);maxU=Math.max(maxU,Math.hypot(ux,uy));error+=Math.hypot(ux-this.u,uy);count++;}
    return {finite,min_density:min,max_density:max,max_lattice_speed:maxU,max_mach:maxU*Math.sqrt(3),mean_velocity_departure:error/count,steps:this.steps,convective_times:this.steps*this.u/this.chord,fluid_cells:count,solid_cells:this.mask.length-count};
  }
}

// Numerical kernels above are unchanged; central permits govern initialization and stepping.
if (!globalThis.UnifiedSimulationBrowser) throw new Error('Central browser simulation adapter is required.');
UnifiedSimulationBrowser.bind(AirflowLBM, 'flow.lbm2d', solver=>solver.w*solver.h);
