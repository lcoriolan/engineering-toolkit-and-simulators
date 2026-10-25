// DESCRIPTION: Genuine three-dimensional D3Q19 BGK GPU flow with volume-atlas textures.
'use strict';
class AirframeLBM{
  /** Allocate a cubic-cell volume. Nineteen post-collision populations use five
   * horizontal RGBA atlas groups; z layers stack vertically. Two passes compute
   * streamed moments then collision, avoiding a requirement for five MRT targets.
   */
  constructor(canvas,grid=[128,64,64]){
    this.canvas=canvas;this.grid=grid;[this.nx,this.ny,this.nz]=grid;this.n=this.nx*this.ny*this.nz;this.u=.04;this.steps=0;this.front=0;
    const gl=this.gl=canvas.getContext('webgl2',{antialias:false,preserveDrawingBuffer:true});
    if(!gl||!gl.getExtension('EXT_color_buffer_float'))throw Error('WebGL2 float render targets unavailable; no flow solver is running.');
    if(this.nx*5>gl.getParameter(gl.MAX_TEXTURE_SIZE)||this.ny*this.nz>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('This graphics backend cannot hold that grid; choose a lower resolution.');
    const info=gl.getExtension('WEBGL_debug_renderer_info');this.renderer=info?gl.getParameter(info.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
    this.resources=[];
    const vertex=`#version 300 es
    void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}`;
    const common=`#version 300 es
    precision highp float;precision highp int;
    uniform sampler2D populations;uniform sampler2D moments;uniform sampler2D obstacle;
    uniform ivec3 grid;uniform float inlet;uniform float tau;
    const ivec3 e[19]=ivec3[19](ivec3(0,0,0),ivec3(1,0,0),ivec3(-1,0,0),ivec3(0,1,0),ivec3(0,-1,0),ivec3(0,0,1),ivec3(0,0,-1),ivec3(1,1,0),ivec3(-1,-1,0),ivec3(1,-1,0),ivec3(-1,1,0),ivec3(1,0,1),ivec3(-1,0,-1),ivec3(1,0,-1),ivec3(-1,0,1),ivec3(0,1,1),ivec3(0,-1,-1),ivec3(0,1,-1),ivec3(0,-1,1));
    const int opp[19]=int[19](0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17);
    float weight(int i){return i==0?1./3.:(i<7?1./18.:1./36.);}
    ivec2 atlas(ivec3 p){p=clamp(p,ivec3(0),grid-1);return ivec2(p.x,p.y+grid.y*p.z);}
    float f(ivec3 p,int i){return texelFetch(populations,atlas(p)+ivec2(grid.x*(i/4),0),0)[i%4];}
    bool solid(ivec3 p){return texelFetch(obstacle,atlas(p),0).r>.5;}
    float eq(int i,float rho,vec3 u){float eu=dot(vec3(e[i]),u);return weight(i)*rho*(1.+3.*eu+4.5*eu*eu-1.5*dot(u,u));}
    // The five non-outlet faces are equilibrium reservoirs; the outlet is a
    // copied distribution. These finite boundaries are not nonreflecting CFD.
    float incoming(ivec3 p,int i){
      if(solid(p))return eq(i,1.,vec3(0));
      if(p.x==0||p.y==0||p.z==0||p.y==grid.y-1||p.z==grid.z-1)return eq(i,1.,vec3(inlet,0,0));
      if(p.x==grid.x-1)return f(p-ivec3(1,0,0),i);
      return solid(p-e[i])?f(p,opp[i]):f(p-e[i],i);
    }
    vec4 macro(ivec3 p){return texelFetch(moments,atlas(p),0);}
    `;
    this.momentProgram=this.program(vertex,common+`
    out vec4 value;
    void main(){ivec2 uv=ivec2(gl_FragCoord.xy);ivec3 p=ivec3(uv.x,uv.y%grid.y,uv.y/grid.y);float rho=0.;vec3 mom=vec3(0);
      for(int i=0;i<19;i++){float q=incoming(p,i);rho+=q;mom+=q*vec3(e[i]);}value=vec4(mom/max(rho,1e-9),rho);}`);
    this.collisionProgram=this.program(vertex,common+`
    out vec4 value;
    void main(){ivec2 uv=ivec2(gl_FragCoord.xy);int pack=uv.x/grid.x;ivec3 p=ivec3(uv.x%grid.x,uv.y%grid.y,uv.y/grid.y);vec4 m=macro(p);vec4 packValue=vec4(0);
      for(int k=0;k<4;k++){int i=pack*4+k;if(i<19){float q=incoming(p,i);packValue[k]=q-(q-eq(i,m.w,m.xyz))/tau;}}value=packValue;}`);
    // The scene is in units of nx cells, preserving physical proportions.
    // Ray marching displays the same occupied cells used by the solver; a
    // selected plane samples the 3D velocity/density field, never a 2D solve.
    this.viewProgram=this.program(vertex,common+`
    uniform vec2 viewport;uniform vec3 eye;uniform vec3 right;uniform vec3 up;uniform vec3 forward;
    uniform float zoom;uniform float pressureScale;uniform int plane;uniform float position;uniform int field;uniform bool showBody;
    out vec4 color;
    vec3 palette(float t){t=clamp(t,0.,1.);return t<.55?mix(vec3(.035,.07,.15),vec3(.05,.48,.67),t/.55):t<.8?mix(vec3(.05,.48,.67),vec3(.4,.87,.7),(t-.55)/.25):mix(vec3(.4,.87,.7),vec3(1,.67,.33),(t-.8)/.2);}
    void main(){vec3 size=vec3(grid)/float(grid.x);vec2 screen=(gl_FragCoord.xy/viewport-.5)*vec2(viewport.x/viewport.y,1.);vec3 ro=eye+(right*screen.x+up*screen.y)*zoom,rd=forward;
      vec3 inv=1./(rd+vec3(1e-9)),a=(vec3(0)-ro)*inv,b=(size-ro)*inv;float enter=max(max(min(a.x,b.x),min(a.y,b.y)),min(a.z,b.z)),leave=min(min(max(a.x,b.x),max(a.y,b.y)),max(a.z,b.z));
      vec3 col=vec3(.045,.075,.10);if(leave<max(enter,0.)){color=vec4(col,1);return;}enter=max(enter,0.);
      float hit=1e9;vec3 hitPoint=vec3(0);
      if(showBody)for(int k=0;k<600;k++){float t=enter+float(k)*.48/float(grid.x);if(t>leave)break;vec3 p=ro+rd*t;ivec3 cell=ivec3(p*float(grid.x));if(solid(cell)){hit=t;hitPoint=p;break;}}
      if(hit<1e8){ivec3 p=ivec3(hitPoint*float(grid.x));vec3 grad=vec3(float(solid(p+ivec3(1,0,0)))-float(solid(p-ivec3(1,0,0))),float(solid(p+ivec3(0,1,0)))-float(solid(p-ivec3(0,1,0))),float(solid(p+ivec3(0,0,1)))-float(solid(p-ivec3(0,0,1))));
        vec3 normal=length(grad)>.01?-normalize(grad):-rd;float light=.5+.5*abs(dot(normal,normalize(vec3(.3,1,.6))));col=vec3(.83,.69,.43)*light;
      }
      int axis=plane;float t=(size[axis]*position-ro[axis])/(rd[axis]+1e-9);vec3 p=ro+rd*t;
      if(t>=enter&&t<=leave&&t<hit){ivec3 cell=ivec3(p*float(grid.x));vec4 m=macro(cell);vec3 flow;
        if(field==0)flow=palette(length(m.xyz)/inlet/1.8);
        else if(field==1){float t=clamp((m.w-1.)/pressureScale,-1.,1.);flow=mix(vec3(.08,.14,.2),t>0.?vec3(1,.55,.25):vec3(.1,.6,1),abs(t));}
        else {vec3 dx=(macro(cell+ivec3(1,0,0)).xyz-macro(cell-ivec3(1,0,0)).xyz)*.5,dy=(macro(cell+ivec3(0,1,0)).xyz-macro(cell-ivec3(0,1,0)).xyz)*.5,dz=(macro(cell+ivec3(0,0,1)).xyz-macro(cell-ivec3(0,0,1)).xyz)*.5;flow=palette(length(vec3(dy.z-dz.y,dz.x-dx.z,dx.y-dy.x))/inlet*5.);}
        col=mix(col,flow,hit<1e8?.68:.9);
      }
      color=vec4(col,1);
    }`);
    // Smooth source-CAD overlay is a separate view of the same assembly.
    // It never changes the conservative solid mask used by either solver.
    this.meshProgram=this.program(`#version 300 es
    precision highp float;
    layout(location=0) in vec3 position;layout(location=1) in vec3 normal;layout(location=2) in vec3 tint;
    uniform vec3 eye;uniform vec3 right;uniform vec3 up;uniform vec3 forward;uniform float zoom;uniform float aspect;uniform float nx;
    out vec3 shade;
    void main(){vec3 d=position/nx-eye;float z=dot(d,forward)/4.;gl_Position=vec4(2.*dot(d,right)/zoom/aspect,2.*dot(d,up)/zoom,2.*z-1.,1.);shade=tint*(.45+.55*abs(dot(normalize(normal),normalize(vec3(.3,1,.6)))));}`,
    `#version 300 es
    precision highp float;in vec3 shade;out vec4 color;void main(){color=vec4(shade,1);}`);
    this.states=[this.target(this.nx*5,this.ny*this.nz),this.target(this.nx*5,this.ny*this.nz)];this.moments=this.target(this.nx,this.ny*this.nz);
    this.maskTexture=this.texture(this.nx,this.ny*this.nz,gl.R8,gl.RED,gl.UNSIGNED_BYTE,null);
  }
  /** Compile and track resources so grid changes can dispose them explicitly. */
  program(vertex,fragment){const gl=this.gl,p=gl.createProgram();for(const [type,src] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));gl.attachShader(p,s);gl.deleteShader(s);}gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));this.resources.push(['Program',p]);return p;}
  /** Nearest sampling preserves lattice values; all atlas dimensions are exact. */
  texture(w,h,internal,format,type,data){const gl=this.gl,t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,internal,w,h,0,format,type,data);this.resources.push(['Texture',t]);return t;}
  /** A single color target works on WebGL2 hardware with four-MRT limits. */
  target(w,h){const gl=this.gl,texture=this.texture(w,h,gl.RGBA32F,gl.RGBA,gl.FLOAT,null),frame=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,frame);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,texture,0);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Incomplete 3D float framebuffer');this.resources.push(['Framebuffer',frame]);return {texture,frame,w,h};}
  /** Reset at the prescribed length-based Re; mesh and mask identity are retained. */
  reset(vox,re=200){if(!Number.isFinite(re)||re<50||re>400)throw Error('Supported length Reynolds range is 50–400');this.vox=vox;this.re=re;this.length=vox.length_cells;this.tau=.5+3*this.u*this.length/re;this.steps=0;this.front=0;
    const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,this.nx,this.ny*this.nz,gl.RED,gl.UNSIGNED_BYTE,vox.mask);
    const e=AirframeLBM.directions,arrays=new Float32Array(this.n*20);
    // Texture rows contain five nx-wide packs, not five complete volumes.
    for(let n=0;n<this.n;n++){const x=n%this.nx,row=Math.floor(n/this.nx),u=vox.mask[n]?0:this.u;for(let i=0;i<19;i++){const eu=e[i][0]*u,weight=i===0?1/3:i<7?1/18:1/36;arrays[(row*this.nx*5+Math.floor(i/4)*this.nx+x)*4+i%4]=weight*(1+3*eu+4.5*eu*eu-1.5*u*u);}}
    for(const state of this.states){gl.bindTexture(gl.TEXTURE_2D,state.texture);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,state.w,state.h,gl.RGBA,gl.FLOAT,arrays);}this.updateMoments();
  }
  /** Bind current population state and metadata; output targets are set separately. */
  bind(p){const gl=this.gl;gl.useProgram(p);[this.states[this.front].texture,this.moments.texture,this.maskTexture].forEach((t,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,t);gl.uniform1i(gl.getUniformLocation(p,['populations','moments','obstacle'][i]),i);});gl.uniform3i(gl.getUniformLocation(p,'grid'),...this.grid);gl.uniform1f(gl.getUniformLocation(p,'inlet'),this.u);gl.uniform1f(gl.getUniformLocation(p,'tau'),this.tau);}
  /** Compute streamed velocity/density. Binding an unused sampler is harmless;
   * the moment shader optimizes it out, so there is no framebuffer feedback.
   */
  updateMoments(){const gl=this.gl;this.bind(this.momentProgram);gl.bindFramebuffer(gl.FRAMEBUFFER,this.moments.frame);gl.viewport(0,0,this.moments.w,this.moments.h);gl.drawArrays(gl.TRIANGLES,0,3);}
  /** Each step includes all x/y/z and diagonal transport; no stacked 2D solves. */
  step(count=1){const gl=this.gl;for(let k=0;k<count;k++){this.updateMoments();this.bind(this.collisionProgram);const out=this.states[1-this.front];gl.bindFramebuffer(gl.FRAMEBUFFER,out.frame);gl.viewport(0,0,out.w,out.h);gl.drawArrays(gl.TRIANGLES,0,3);this.front=1-this.front;this.steps++;}}
  /** Orbit a volume and an arbitrary orthogonal slice, using the actual 3D field. */
  draw(options={}){const gl=this.gl,p=this.viewProgram;this.bind(p);gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,this.canvas.width,this.canvas.height);
    const yaw=options.yaw??.35,pitch=options.pitch??.7,dir=[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)],right=[Math.cos(yaw),0,-Math.sin(yaw)],up=[-Math.sin(yaw)*Math.sin(pitch),Math.cos(pitch),-Math.cos(yaw)*Math.sin(pitch)],center=this.grid.map(v=>v/this.nx/2);
    for(const [name,value] of [['eye',center.map((v,k)=>v+dir[k]*2)],['right',right],['up',up],['forward',dir.map(v=>-v)]])gl.uniform3f(gl.getUniformLocation(p,name),...value);
    gl.uniform1f(gl.getUniformLocation(p,'pressureScale'),options.pressureScale??.015);gl.uniform2f(gl.getUniformLocation(p,'viewport'),this.canvas.width,this.canvas.height);gl.uniform1f(gl.getUniformLocation(p,'zoom'),options.zoom??1.05);gl.uniform1i(gl.getUniformLocation(p,'plane'),options.plane??1);gl.uniform1f(gl.getUniformLocation(p,'position'),options.position??.5);gl.uniform1i(gl.getUniformLocation(p,'field'),options.field??0);gl.uniform1i(gl.getUniformLocation(p,'showBody'),options.showBody===false||options.geometry==='cad'?0:1);gl.drawArrays(gl.TRIANGLES,0,3);
    if(options.geometry==='cad'&&options.showBody!==false&&this.meshBuffer){
      const mesh=this.meshProgram;gl.useProgram(mesh);for(const [name,value] of [['eye',center.map((v,k)=>v+dir[k]*2)],['right',right],['up',up],['forward',dir.map(v=>-v)]])gl.uniform3f(gl.getUniformLocation(mesh,name),...value);
      gl.uniform1f(gl.getUniformLocation(mesh,'zoom'),options.zoom??1.05);gl.uniform1f(gl.getUniformLocation(mesh,'aspect'),this.canvas.width/this.canvas.height);gl.uniform1f(gl.getUniformLocation(mesh,'nx'),this.nx);
      gl.clear(gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.bindBuffer(gl.ARRAY_BUFFER,this.meshBuffer);for(let i=0;i<3;i++){gl.enableVertexAttribArray(i);gl.vertexAttribPointer(i,3,gl.FLOAT,false,36,i*12);}
      gl.drawArrays(gl.TRIANGLES,0,this.meshCount);for(let i=0;i<3;i++)gl.disableVertexAttribArray(i);gl.disable(gl.DEPTH_TEST);
    }
  }
  /** Upload complete source triangles for visual comparison with the stepped
   * boundary. Flat shading retains real facets; buffers are released on rebuild.
   */
  setMesh(parts){const values=[];
    for(const part of parts)for(const ids of part.triangles){const [a,b,c]=ids.map(i=>part.vertices[i]),u=b.map((v,k)=>v-a[k]),v=c.map((q,k)=>q-a[k]),n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],length=Math.hypot(...n)||1,normal=n.map(q=>q/length),tint=/fin|wing/i.test(part.name)?[.94,.63,.28]:[.54,.72,.77];for(const point of [a,b,c])values.push(...point,...normal,...tint);}
    const gl=this.gl;if(this.meshBuffer)gl.deleteBuffer(this.meshBuffer);this.meshBuffer=gl.createBuffer();this.resources.push(['Buffer',this.meshBuffer]);gl.bindBuffer(gl.ARRAY_BUFFER,this.meshBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.STATIC_DRAW);this.meshCount=values.length/9;
  }
  /** Read current streamed moments for integrity checks/export, not validated loads. */
  readMoments(){this.updateMoments();const gl=this.gl,out=new Float32Array(this.n*4);gl.bindFramebuffer(gl.FRAMEBUFFER,this.moments.frame);gl.readPixels(0,0,this.nx,this.ny*this.nz,gl.RGBA,gl.FLOAT,out);return out;}
  /** Read populations solely for a scalar-reference numerical test. */
  readPopulations(){const gl=this.gl,s=this.states[this.front],out=new Float32Array(this.n*20);gl.bindFramebuffer(gl.FRAMEBUFFER,s.frame);gl.readPixels(0,0,s.w,s.h,gl.RGBA,gl.FLOAT,out);return out;}
  /** Density and numerical Mach catch instability, not errors in aero prediction. */
  diagnostics(){const a=this.readMoments();let lo=Infinity,hi=-Infinity,maxU=0,transverse=0,departure=0,count=0,finite=true;for(let n=0;n<this.n;n++){if(this.vox.mask[n])continue;const x=a[n*4],y=a[n*4+1],z=a[n*4+2],rho=a[n*4+3];finite&&=Number.isFinite(x)&&Number.isFinite(y)&&Number.isFinite(z)&&Number.isFinite(rho);lo=Math.min(lo,rho);hi=Math.max(hi,rho);maxU=Math.max(maxU,Math.hypot(x,y,z));transverse+=Math.hypot(y,z);departure+=Math.hypot(x-this.u,y,z);count++;}return {finite,min_density:lo,max_density:hi,max_mach:maxU*Math.sqrt(3),mean_transverse_speed:transverse/count,mean_velocity_departure:departure/count,steps:this.steps,convective_times:this.steps*this.u/this.length,fluid_cells:count,solid_cells:this.n-count};}
  /** Release GPU allocations before changing grids; retain the browser context. */
  dispose(){for(const [type,obj] of this.resources)this.gl['delete'+type](obj);this.resources=[];}
}
AirframeLBM.directions=[[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]];

// Numerical kernels above are unchanged; central permits govern initialization and stepping.
if (!globalThis.UnifiedSimulationBrowser) throw new Error('Central browser simulation adapter is required.');
UnifiedSimulationBrowser.bind(AirframeLBM, 'flow.lbm3d', solver=>solver.n);
