// --- Global State & DOM Elements ---
let isRendering = true; // Start rendering by default
let monacoEditor;
let currentKernel = ""; // The currently *compiled* kernel
let hasHWA = false; // Hardware acceleration check
let isMobile = false;
let autoApplyTimeout = null;

let frameCount = 0;
let totalFrames = 0;
let fps = 0;
let fpsLastUpdate = performance.now();
let renderStartTime = performance.now();
let frameTimes = [];
const maxSamples = 600;
let lastFrameTimestamp = performance.now();

let rotationSpeed = 50;
let cx, cy;
let glposition, glright, glforward, glup, glorigin, glx, gly, gllen;
let canvas, gl;
let md = 0, mx, my;
let mx1 = 0, my1 = 0, lasttimen = 0;
let ml = 0, mr = 0, mm = 0;
let len = 2.6;
let ang1 = 2.8;
let ang2 = 0.4;
let cenx = 0.0, ceny = 0.0, cenz = 0.0;

let vertshader, fragshader, shaderProgram;

// Base shader code (the part that doesn't change)
const FSHADER_SOURCE =
`#version 100
#define PI 3.14159265358979324
#define M_L 0.3819660113
#define M_R 0.6180339887
#define MAXR 8
#define SOLVER 8
precision highp float;
float kernal(vec3 ver);
uniform vec3 right, forward, up, origin;
varying vec3 dir, localdir;
uniform float len;
vec3 ver;
int sign;
float v, v1, v2;
float r1, r2, r3, r4, m1, m2, m3, m4;
vec3 n, reflect;
const float step = 0.002;
vec3 color;
void main() {
   color.r=0.0;
   color.g=0.0;
   color.b=0.0;
   sign=0;
   v1 = kernal(origin + dir * (step*len));
   v2 = kernal(origin);
   for (int k = 2; k < 1002; k++) {
      ver = origin + dir * (step*len*float(k));
      v = kernal(ver);
      if (v > 0.0 && v1 < 0.0) {
         r1 = step * len*float(k - 1);
         r2 = step * len*float(k);
         m1 = kernal(origin + dir * r1);
         m2 = kernal(origin + dir * r2);
         for (int l = 0; l < SOLVER; l++) {
            r3 = r1 * 0.5 + r2 * 0.5;
            m3 = kernal(origin + dir * r3);
            if (m3 > 0.0) {
               r2 = r3;
               m2 = m3;
            }
            else {
               r1 = r3;
               m1 = m3;
            }
         }
         if (r3 < 2.0 * len) {
               sign=1;
            break;
         }
      }
      if (v < v1&&v1>v2&&v1 < 0.0 && (v1*2.0 > v || v1 * 2.0 > v2)) {
         r1 = step * len*float(k - 2);
         r2 = step * len*(float(k) - 2.0 + 2.0*M_L);
         r3 = step * len*(float(k) - 2.0 + 2.0*M_R);
         r4 = step * len*float(k);
         m2 = kernal(origin + dir * r2);
         m3 = kernal(origin + dir * r3);
         for (int l = 0; l < MAXR; l++) {
            if (m2 > m3) {
               r4 = r3;
               r3 = r2;
               r2 = r4 * M_L + r1 * M_R;
               m3 = m2;
               m2 = kernal(origin + dir * r2);
            }
            else {
               r1 = r2;
               r2 = r3;
               r3 = r4 * M_R + r1 * M_L;
               m2 = m3;
               m3 = kernal(origin + dir * r3);
            }
         }
         if (m2 > 0.0) {
            r1 = step * len*float(k - 2);
            r2 = r2;
            m1 = kernal(origin + dir * r1);
            m2 = kernal(origin + dir * r2);
            for (int l = 0; l < SOLVER; l++) {
               r3 = r1 * 0.5 + r2 * 0.5;
               m3 = kernal(origin + dir * r3);
               if (m3 > 0.0) {
                  r2 = r3;
                  m2 = m3;
               }
               else {
                  r1 = r3;
                  m1 = m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
                   sign=1;
               break;
            }
         }
         else if (m3 > 0.0) {
            r1 = step * len*float(k - 2);
            r2 = r3;
            m1 = kernal(origin + dir * r1);
            m2 = kernal(origin + dir * r2);
            for (int l = 0; l < SOLVER; l++) {
               r3 = r1 * 0.5 + r2 * 0.5;
               m3 = kernal(origin + dir * r3);
               if (m3 > 0.0) {
                  r2 = r3;
                  m2 = m3;
               }
               else {
                  r1 = r3;
                  m1 = m3;
               }
            }
            if (r3 < 2.0 * len&&r3> step*len) {
                   sign=1;
               break;
            }
         }
      }
      v2 = v1;
      v1 = v;
   }
   if (sign==1) {
      ver = origin + dir*r3 ;
      r1=ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      n.x = kernal(ver - right * (r3*0.00025)) - kernal(ver + right * (r3*0.00025));
      n.y = kernal(ver - up * (r3*0.00025)) - kernal(ver + up * (r3*0.00025));
      n.z = kernal(ver + forward * (r3*0.00025)) - kernal(ver - forward * (r3*0.00025));
      r3 = n.x*n.x+n.y*n.y+n.z*n.z;
      n = n * (1.0 / sqrt(r3));
      ver = localdir;
      r3 = ver.x*ver.x+ver.y*ver.y+ver.z*ver.z;
      ver = ver * (1.0 / sqrt(r3));
      reflect = n * (-2.0*dot(ver, n)) + ver;
      r3 = reflect.x*0.276+reflect.y*0.920+reflect.z*0.276;
      r4 = n.x*0.276+n.y*0.920+n.z*0.276;
      r3 = max(0.0,r3);
      r3 = r3 * r3*r3*r3;
      r3 = r3 * 0.45 + r4 * 0.25 + 0.3;
      n.x = sin(r1*10.0)*0.5+0.5;
      n.y = sin(r1*10.0+2.05)*0.5+0.5;
      n.z = sin(r1*10.0-2.05)*0.5+0.5;
      color = n*r3;
   }
   gl_FragColor = vec4(color.x, color.y, color.z, 1.0);
}`;

// Default kernel to start with
const DEFAULT_KERNEL = `float boxDist(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0));
}
  
float kernal(vec3 ver) {
    return 0.0000001 - boxDist(ver, vec3(1.0));
}`;

// Default presets
const defaultPresets = {
    "Very Simple Box": `float boxDist(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0));
}
  
float kernal(vec3 ver) {
    return 0.0000001 - boxDist(ver, vec3(1.0));
}`,
    "Very Simple": `float boxDist(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0));
}

float kernal(vec3 ver) {
    return 1.0 - boxDist(ver, vec3(1.0));
}`,
    "Simple": `float boxDist(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0));
}

float kernal(vec3 ver) {
    vec3 a = ver;
    float b = 0.0;
    for (int i = 0; i < 2; i++) {
        b = dot(a, a);
        if (b > 36.0) break;
        a = a * clamp(1.1 - b * 0.04, 0.6, 1.1) + ver * 0.25;
    }
    return 1.0 - boxDist(a, vec3(1.0));
}`,
    "Base": `float kernal(vec3 ver) {
    vec3 a;
    float b,c,d,e;
    a=ver;
    for(int i=0;i<5;i++){
        b=length(a);
        c=atan(a.y,a.x)*8.0;
        e=1.0/b;
        d=acos(a.z/b)*8.0;
        b=pow(b,8.0);
        a=vec3(b*sin(d)*cos(c),b*sin(d)*sin(c),b*cos(d))+ver;
        if(b>6.0){
            break;
        }
    }   
    return 4.0-a.x*a.x-a.y*a.y-a.z*a.z;
}`,
    "Hard Fractal": `float kernal(vec3 ver) {
    vec3 a = ver;
    float b, c, d, e, f, g;
  
    for (int i = 0; i < 25; i++) {
        b = length(a) + 0.001;
        c = atan(a.y, a.x) * 12.0 + sin(a.z * 5.0);
        d = acos(clamp(a.z / b, -1.0, 1.0)) * 12.0 + cos(a.x * 5.0);
        e = min(pow(b, 4.0 + sin(float(i)) * 0.5), 20.0);
        f = sin(c * 0.5 + sin(d * 0.25 + cos(b * 2.0)));
        g = cos(d * 0.5 + cos(c * 0.25 + sin(b * 2.0)));
  
        a = vec3(
            e * sin(d) * cos(c) + f * 0.1,
            e * sin(d) * sin(c) + g * 0.1,
            e * cos(d) + f * g * 0.05
        ) + ver * (0.3 + 0.3 * sin(float(i)));
  
        if (e > 10.0) break;
    }
  
    float dist = 4.0 - dot(a, a);
    return dist;
}`,
    "Extreme Fractal": `float kernal(vec3 ver) {
    vec3 a = ver;
    float b, c, d, e, f, g, h;
  
    for (int i = 0; i < 80; i++) {
        b = length(a) + 0.0001;
  
        c = atan(a.y, a.x) * 24.0 + sin(a.z * 9.0 + cos(b * 2.0));
        d = acos(clamp(a.z / b, -1.0, 1.0)) * 24.0 + cos(a.x * 9.0 + sin(b * 3.0));
  
        e = pow(b, 9.0 + sin(float(i * 2)) * 1.5);
  
        f = sin(c * 1.5 + sin(d * 0.75 + cos(b * 3.0)));
        g = cos(d * 1.5 + cos(c * 0.75 + sin(b * 3.0)));
  
        h = sin(dot(a, a) * 5.0 + float(i) * 0.25);
  
        a = vec3(
            e * sin(d) * cos(c) + f * g * h * 0.4,
            e * sin(d) * sin(c) + f * g * h * 0.4,
            e * cos(d) + f * g * h * 0.4
        ) + ver * (0.8 + 0.6 * sin(float(i) + dot(ver, a)));
    }
  
    float dist = 0.5 - dot(a, a);
    float sparkle = sin(dot(a, a) * 20.0) * 0.1;
    return dist + sparkle;
}`
};

// DOM Element references
const toggleRenderBtn = document.getElementById("toggleRender");
const renderIcon = document.getElementById("render-icon");
const renderText = document.getElementById("render-text");
const presetList = document.getElementById("presetList");
const savePresetBtn = document.getElementById("savePresetBtn");
const deletePresetBtn = document.getElementById("deletePresetBtn");
const speedSlider = document.getElementById("rotationSpeed");
const speedValue = document.getElementById("speedValue");
const fpsCounter = document.getElementById("fpsCounter");
const panel = document.getElementById("right-panel");
const togglePanelBtn = document.getElementById("togglePanelBtn");
const applyBtn = document.getElementById("apply");
const cancelBtn = document.getElementById("cancel");
const errorConsole = document.getElementById("error-console");
const autoApplyCheckbox = document.getElementById("autoApply");

// Modals
const saveModal = document.getElementById("saveModal");
const deleteModal = document.getElementById("deleteModal");
const helpModal = document.getElementById("helpModal");
const presetNameInput = document.getElementById("presetNameInput");
const presetToDeleteName = document.getElementById("presetToDeleteName");


// --- Utility Functions ---

/**
 * Shows a toast notification.
 * @param {string} message The message to display.
 * @param {'info' | 'success' | 'error'} type The type of toast.
 * @param {number} duration How long to show the toast (in ms).
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = '';
    if (type === 'success') {
        icon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else if (type === 'error') {
        icon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    }
    
    toast.innerHTML = `${icon}<span>${message}</span>`;
    container.appendChild(toast);
    
    // Animate in
    setTimeout(() => toast.classList.add('show'), 10);

    // Animate out
    setTimeout(() => {
        toast.classList.remove('show');
        toast.addEventListener('transitionend', () => toast.remove());
    }, duration);
}

/**
 * Creates a debounced version of a function.
 * @param {Function} func The function to debounce.
 * @param {number} delay The delay in milliseconds.
 */
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
}

/**
 * Checks for WebGL Hardware Acceleration.
 */
function checkHWA() {
    try {
        const test = (force) => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d", { willReadFrequently: force });
            ctx.moveTo(0, 0);
            ctx.lineTo(120, 121);
            ctx.stroke();
            return ctx.getImageData(0, 0, 200, 200).data.join();
        };
        hasHWA = test(true) !== test(false);
    } catch (e) {
        hasHWA = false; // Failed to test, assume no HWA
    }
    if (!hasHWA) {
        showToast("Hardware Acceleration is disabled. Performance may be affected.", "error", 5000);
    }
}

/**
 * Detects if the user is on a mobile device.
 */
function detectMobile() {
    isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}


// --- Preset Management ---

/**
 * Gets presets from localStorage.
 */
function getPresets() {
    let presets;
    try {
        presets = JSON.parse(localStorage.getItem("presets"));
    } catch (e) {
        presets = null;
    }
    
    if (!presets || typeof presets !== 'object' || Array.isArray(presets)) {
        presets = defaultPresets;
        setPresets(presets);
    }
    
    // Ensure default presets are always there
    let needsUpdate = false;
    for (const key in defaultPresets) {
        if (!presets[key]) {
            presets[key] = defaultPresets[key];
            needsUpdate = true;
        }
    }
    if (needsUpdate) {
        setPresets(presets);
    }
    
    return presets;
}

/**
 * Saves presets to localStorage.
 * @param {object} presets The presets object to save.
 */
function setPresets(presets) {
    try {
        localStorage.setItem("presets", JSON.stringify(presets));
    } catch (e) {
        showToast("Error saving presets. Storage may be full.", "error");
    }
}

/**
 * Updates the <select> dropdown list with presets.
 */
function updatePresetList() {
    const presets = getPresets();
    presetList.innerHTML = `<option disabled selected>Select a preset to apply</option>`;
    for (const name in presets) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        presetList.appendChild(option);
    }
}


// --- WebGL & Rendering ---

/**
 * Applies the kernel from the editor to the WebGL shader.
 */
function applyKernel() {
    if (!hasHWA) {
        showToast("Cannot apply shader: Hardware Acceleration is disabled.", "error");
        return;
    }
    
    applyBtn.classList.remove("is-success", "is-error");
    applyBtn.classList.add("is-loading");
    applyBtn.disabled = true;

    // Use a short timeout to allow the UI to update
    setTimeout(() => {
        const kernelCode = monacoEditor.getValue();
        gl.shaderSource(fragshader, FSHADER_SOURCE + kernelCode);
        gl.compileShader(fragshader);
        const infof = gl.getShaderInfoLog(fragshader);
        
        gl.linkProgram(shaderProgram);
        
        if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(shaderProgram);
            errorConsole.textContent = infof + info;
            errorConsole.classList.add("visible");
            showToast("Shader failed to compile!", "error");
            
            applyBtn.classList.remove("is-loading");
            applyBtn.classList.add("is-error");
            applyBtn.disabled = false;
        } else {
            errorConsole.classList.remove("visible");
            currentKernel = kernelCode; // Save this as the last known good kernel
            
            // Re-bind uniforms
            glposition = gl.getAttribLocation(shaderProgram, 'position');
            glright = gl.getUniformLocation(shaderProgram, 'right');
            glforward = gl.getUniformLocation(shaderProgram, 'forward');
            glup = gl.getUniformLocation(shaderProgram, 'up');
            glorigin = gl.getUniformLocation(shaderProgram, 'origin');
            glx = gl.getUniformLocation(shaderProgram, 'x');
            gly = gl.getUniformLocation(shaderProgram, 'y');
            gllen = gl.getUniformLocation(shaderProgram, 'len');
            
            // Reset FPS counters
            totalFrames = 0;
            renderStartTime = performance.now();
            frameTimes = [];
            
            showToast("Shader applied successfully!", "success");
            applyBtn.classList.remove("is-loading");
            applyBtn.classList.add("is-success");
            applyBtn.disabled = false;

            setTimeout(() => {
                applyBtn.classList.remove("is-success");
            }, 1500);
        }
    }, 50);
}

/**
 * Updates the FPS counter.
 */
function updateFPS() {
    const now = performance.now();
    const delta = now - lastFrameTimestamp;
    lastFrameTimestamp = now;

    frameTimes.push(delta);
    if (frameTimes.length > maxSamples) {
        frameTimes.shift();
    }

    frameCount++;
    totalFrames++;

    const elapsed = now - fpsLastUpdate;

    if (elapsed >= 500) { // Update counter twice a second
        fps = Math.round((frameCount * 1000) / elapsed);

        const totalTime = (now - renderStartTime) / 1000;
        const avgFps = Math.round(totalFrames / totalTime);

        // 1% Low FPS
        const sorted = [...frameTimes].sort((a, b) => b - a); // longest frame times first
        const onePercentIndex = Math.floor(sorted.length * 0.01);
        const onePercentLow = sorted[onePercentIndex] || sorted[sorted.length - 1] || 1;
        const onePercentFps = Math.round(1000 / onePercentLow);

        fpsCounter.innerHTML = `FPS: <span>${fps}</span> | Avg: <span>${avgFps}</span> | 1% Low: <span>${onePercentFps}</span>`;

        fpsLastUpdate = now;
        frameCount = 0;
    }
}

/**
 * Main drawing function.
 */
function draw() {
    updateFPS();
    if (!isRendering) return;

    if (isRendering && rotationSpeed > 0) {
        ang1 += rotationSpeed / 5000;
    }

    gl.uniform1f(glx, cx * 2.0 / (cx + cy));
    gl.uniform1f(gly, cy * 2.0 / (cx + cy));
    gl.uniform1f(gllen, len);
    gl.uniform3f(glorigin, len * Math.cos(ang1) * Math.cos(ang2) + cenx, len * Math.sin(ang2) + ceny, len * Math.sin(ang1) * Math.cos(ang2) + cenz);
    gl.uniform3f(glright, Math.sin(ang1), 0, -Math.cos(ang1));
    gl.uniform3f(glup, -Math.sin(ang2) * Math.cos(ang1), Math.cos(ang2), -Math.sin(ang2) * Math.sin(ang1));
    gl.uniform3f(glforward, -Math.cos(ang1) * Math.cos(ang2), -Math.sin(ang2), -Math.sin(ang1) * Math.cos(ang2));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.finish();
}

/**
 * The main render loop.
 */
function onTimer() {
    draw();
    window.requestAnimationFrame(onTimer);
}

/**
 * Resizes the canvas and viewport.
 */
function resizeCanvas() {
    const renderArea = document.querySelector('.render-area');
    cx = renderArea.clientWidth;
    cy = renderArea.clientHeight;
    if (cy === 0) return; 
    
    const baseHeight = 1024;
    const aspectRatio = cx / cy;
    
    const bufferWidth = Math.round(baseHeight * aspectRatio);
    const bufferHeight = baseHeight;

    canvas.width = bufferWidth;
    canvas.height = bufferHeight;

    canvas.style.width = `${cx}px`;
    canvas.style.height = `${cy}px`;

    gl.viewport(0, 0, bufferWidth, bufferHeight);
    
    if (monacoEditor) {
        monacoEditor.layout();
    }
}

// --- Initialization Functions ---

/**
 * Initializes the Monaco Editor.
 */
function initMonaco() {
    if (typeof require === 'undefined') {
        setTimeout(initMonaco, 100);
        return;
    }

    require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' }});
    require(['vs/editor/editor.main'], function(monaco) {
        monaco.languages.register({ id: 'glsl' });

        monaco.languages.setLanguageConfiguration('glsl', {
            comments: {
                lineComment: '//',
                blockComment: ['/*', '*/']
            },
            brackets: [
                ['{', '}'],
                ['[', ']'],
                ['(', ')']
            ],
            autoClosingPairs: [
                { open: '{', close: '}' },
                { open: '[', close: ']' },
                { open: '(', close: ')' },
                { open: '"', close: '"', notIn: ['string'] },
                { open: "'", close: "'", notIn: ['string', 'comment'] }
            ],
            surroundingPairs: [
                { open: '{', close: '}' },
                { open: '[', close: ']' },
                { open: '(', close: ')' },
                { open: '"', close: '"' },
                { open: "'", close: "'" }
            ]
        });

        monaco.languages.setMonarchTokensProvider('glsl', {
            keywords: [
                'for', 'if', 'else', 'while', 'do', 'return', 'break', 'continue', 'struct',
                'const', 'uniform', 'varying', 'attribute', 'layout', 'in', 'out', 'inout',
                'precision', 'highp', 'mediump', 'lowp', 'discard'
            ],
            types: [
                'void', 'bool', 'int', 'uint', 'float', 'double',
                'vec2', 'vec3', 'vec4', 'bvec2', 'bvec3', 'bvec4',
                'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4',
                'mat2', 'mat3', 'mat4', 'mat2x2', 'mat2x3', 'mat2x4',
                'mat3x2', 'mat3x3', 'mat3x4', 'mat4x2', 'mat4x3', 'mat4x4',
                'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DShadow'
            ],
            builtInFunctions: [
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
                'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
                'abs', 'sign', 'floor', 'ceil', 'fract', 'mod', 'min', 'max', 'clamp', 'mix', 'step', 'smoothstep',
                'length', 'distance', 'dot', 'cross', 'normalize', 'faceforward', 'reflect', 'refract',
                'matrixCompMult', 'outerProduct', 'transpose', 'determinant', 'inverse',
                'lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual', 'equal', 'notEqual', 'any', 'all', 'not',
                'texture2D', 'textureCube', 'texture', 'textureProj', 'textureLod', 'textureOffset', 'texelFetch'
            ],
            builtInVariables: [
                'gl_Position', 'gl_FragColor', 'gl_FragCoord', 'gl_PointCoord', 'gl_PointSize', 'gl_FragData',
                'gl_FrontFacing', 'gl_VertexID', 'gl_InstanceID'
            ],
            operators: [
                '=', '>', '<', '!', '~', '?', ':', '==', '<=', '>=', '!=', '&&', '||', '++', '--',
                '+', '-', '*', '/', '&', '|', '^', '%', '<<', '>>', '+=', '-=', '*=', '/=',
                '&=', '|=', '^=', '%=', '<<=', '>>='
            ],

            symbols:  /[=><!~?:&|+\-*\/\^%]+/,

            tokenizer: {
                root: [
                    [/[a-zA-Z_]\w*(?=\s*\()/, 'entity.name.function'],
                    [/[a-zA-Z_]\w*/, {
                        cases: {
                            '@keywords': 'keyword',
                            '@types': 'type.identifier',
                            '@builtInFunctions': 'keyword.function',
                            '@builtInVariables': 'variable.predefined',
                            '@default': 'identifier'
                        }
                    }],
                    { include: '@whitespace' },
                    [/^#\s*[a-zA-Z_]\w*/, 'keyword.directive'],
                    [/[{}()\[\]]/, '@brackets'],

                    [/@symbols/, {
                        cases: {
                            '@operators': 'operator',
                            '@default': ''
                        }
                    }],
                    // numbers
                    [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
                    [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                    [/\d+/, 'number'],
                    // delimiter
                    [/[;,]/, 'delimiter'],
                    // strings
                    [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-teminated string
                    [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],
                ],
                comment: [
                    [/[^\/*]+/, 'comment'],
                    [/\*\//, 'comment', '@pop'],
                    [/[\/*]/, 'comment']
                ],
                string: [
                    [/[^\\"]+/, 'string'],
                    [/\\./, 'string.escape.invalid'],
                    [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }]
                ],
                whitespace: [
                    [/[ \t\r\n]+/, 'white'],
                    [/\/\*/, 'comment', '@comment'],
                    [/\/\/.*$/, 'comment'],
                ],
            }
        });

        monacoEditor = monaco.editor.create(document.getElementById('editor-container'), {
            value: DEFAULT_KERNEL,
            language: 'glsl',
            theme: 'vs-dark',
            automaticLayout: true,
            minimap: { enabled: false },
            roundedSelection: true,
            scrollBeyondLastLine: false,
        });
        
        currentKernel = DEFAULT_KERNEL;
        
        monacoEditor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyQ, function() {
            applyKernel();
        });
        
        monacoEditor.onDidChangeModelContent(() => {
            if (autoApplyCheckbox.checked) {
                if (autoApplyTimeout) clearTimeout(autoApplyTimeout);
                autoApplyTimeout = setTimeout(applyKernel, 750);
            }
        });

        applyKernel();

        window.requestAnimationFrame(onTimer);

        applyBtn.disabled = false;
        cancelBtn.disabled = false;
        savePresetBtn.disabled = false;
        deletePresetBtn.disabled = false;
        presetList.disabled = false;
        autoApplyCheckbox.disabled = false;
    });
}

/**
 * Initializes all WebGL components.
 */
function initGL() {
    canvas = document.getElementById('c1');
    gl = canvas.getContext('webgl');
    if (!gl) {
        showToast("WebGL is not supported by your browser!", "error", 10000);
        return;
    }

    const positions = [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
    const VSHADER_SOURCE =
        `#version 100
        precision highp float;
        attribute vec4 position;
        varying vec3 dir, localdir;
        uniform vec3 right, forward, up, origin;
        uniform float x,y;
        void main() {
           gl_Position = position; 
           dir = forward + right * position.x*x + up * position.y*y;
           localdir.x = position.x*x;
           localdir.y = position.y*y;
           localdir.z = -1.0;
        } `;
        
    vertshader = gl.createShader(gl.VERTEX_SHADER);
    fragshader = gl.createShader(gl.FRAGMENT_SHADER);
    shaderProgram = gl.createProgram();
    
    gl.shaderSource(vertshader, VSHADER_SOURCE);
    gl.compileShader(vertshader);
    
    // Initial compile with default kernel
    gl.shaderSource(fragshader, FSHADER_SOURCE + DEFAULT_KERNEL);
    gl.compileShader(fragshader);
    
    gl.attachShader(shaderProgram, vertshader);
    gl.attachShader(shaderProgram, fragshader);
    gl.linkProgram(shaderProgram);
    gl.useProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        showToast("Initial shader compile failed. Check console.", "error");
        console.error("Initial shader compile failed:", gl.getShaderInfoLog(fragshader), gl.getProgramInfoLog(shaderProgram));
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.vertexAttribPointer(glposition, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(glposition);

    resizeCanvas();
}

/**
 * Initializes all general event listeners.
 */
function initEventListeners() {
    applyBtn.disabled = true;
    cancelBtn.disabled = true;
    savePresetBtn.disabled = true;
    deletePresetBtn.disabled = true;
    presetList.disabled = true;
    autoApplyCheckbox.disabled = true;

    // Panel Collapse
    togglePanelBtn.addEventListener("click", () => {
        panel.classList.toggle("collapsed");
        togglePanelBtn.classList.toggle("collapsed", panel.classList.contains("collapsed"));
        // A short delay to allow CSS transitions to start
        setTimeout(resizeCanvas, 50);
    });

    // Panel Resize (Desktop)
    const handle = document.querySelector('.resize-handle');
    if (handle) {
        handle.addEventListener('mousedown', function (e) {
            e.preventDefault();
            document.body.classList.add('is-resizing');
            panel.style.transition = 'none';
            const startX = e.clientX;
            const startWidth = parseInt(window.getComputedStyle(panel).width, 10);

            function doDrag(e) {
                const targetWidth = startWidth - (e.clientX - startX);
                const minWidth = 300;
                const maxWidth = window.innerWidth * 0.9;
                panel.style.width = `${Math.max(minWidth, Math.min(targetWidth, maxWidth))}px`;
                resizeCanvas();
            }
            function stopDrag() {
                document.body.classList.remove('is-resizing');
                localStorage.setItem('panelWidth', panel.style.width);
                document.removeEventListener('mousemove', doDrag);
                document.removeEventListener('mouseup', stopDrag);
                resizeCanvas();
            }
            document.addEventListener('mousemove', doDrag);
            document.addEventListener('mouseup', stopDrag);
        });
    }

    // Render Toggle
    const playIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
    const pauseIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
    
    toggleRenderBtn.addEventListener("click", () => {
        isRendering = !isRendering;
        if (isRendering) {
            renderText.textContent = "Pause Rendering";
            renderIcon.innerHTML = pauseIcon;
        } else {
            renderText.textContent = "Start Rendering";
            renderIcon.innerHTML = playIcon;
        }
    });

    // Rotation Slider
    speedSlider.addEventListener("input", () => {
        rotationSpeed = parseInt(speedSlider.value, 10);
        speedValue.textContent = rotationSpeed;
    });

    // Shader Buttons
    applyBtn.addEventListener("click", applyKernel);
    cancelBtn.addEventListener("click", () => {
        monacoEditor.setValue(currentKernel);
        errorConsole.classList.remove("visible");
        showToast("Editor reset to last applied kernel.", "info");
    });

    // Preset Controls
    presetList.addEventListener("change", () => {
        const name = presetList.value;
        const presets = getPresets();
        if (presets[name]) {
            monacoEditor.setValue(presets[name]);
            if (autoApplyCheckbox.checked) {
                applyKernel();
            } else {
                showToast("Preset loaded into editor. Hit 'Apply'.", "info");
            }
        }
    });

    // Modal Triggers
    savePresetBtn.addEventListener("click", () => {
        presetNameInput.value = "";
        saveModal.classList.add("visible");
        presetNameInput.focus();
    });

    deletePresetBtn.addEventListener("click", () => {
        const name = presetList.value;
        if (!name || presetList.selectedIndex === 0) {
            showToast("Select a preset to delete first.", "error");
            return;
        }
        presetToDeleteName.textContent = name;
        deleteModal.classList.add("visible");
    });

    document.getElementById("helpBtn").addEventListener("click", () => {
        helpModal.classList.add("visible");
    });

    // Modal Close Buttons
    document.getElementById("cancelSaveBtn").addEventListener("click", () => saveModal.classList.remove("visible"));
    document.getElementById("cancelDeleteBtn").addEventListener("click", () => deleteModal.classList.remove("visible"));
    document.getElementById("closeHelpBtn").addEventListener("click", () => helpModal.classList.remove("visible"));
    
    // Modal Confirm Actions
    document.getElementById("confirmSaveBtn").addEventListener("click", () => {
        const name = presetNameInput.value.trim();
        if (!name) {
            showToast("Please enter a preset name.", "error");
            return;
        }
        const presets = getPresets();
        presets[name] = monacoEditor.getValue();
        setPresets(presets);
        updatePresetList();
        presetList.value = name;
        showToast(`Preset "${name}" saved!`, "success");
        saveModal.classList.remove("visible");
    });
    
    document.getElementById("confirmDeleteBtn").addEventListener("click", () => {
        const name = presetList.value;
        const presets = getPresets();
        if (presets[name]) {
            delete presets[name];
            setPresets(presets);
            updatePresetList();
            monacoEditor.setValue(DEFAULT_KERNEL);
            showToast(`Preset "${name}" deleted.`, "success");
        } else {
            showToast("Could not find preset to delete.", "error");
        }
        deleteModal.classList.remove("visible");
    });

    // Window Resize
    window.addEventListener("resize", debounce(resizeCanvas, 100));
}

/**
 * Initializes all canvas mouse/touch controls.
 */
function initCanvasControls() {
    // --- Mouse Controls ---
    canvas.addEventListener("mousedown", (ev) => {
        const oEvent = ev || event;
        if (oEvent.button == 0) { // Left-click
            ml = 1;
            mm = 0;
            if (isRendering) rotationSpeed = 0; // Stop rotation
        }
        if (oEvent.button == 2) { // Right-click
            mr = 1;
            mm = 0;
        }
        mx = oEvent.clientX;
        my = oEvent.clientY;
    }, false);

    canvas.addEventListener("mouseup", (ev) => {
        const oEvent = ev || event;
        if (oEvent.button == 0) {
            ml = 0;
            if (isRendering) rotationSpeed = parseInt(speedSlider.value, 10); // Resume rotation
        }
        if (oEvent.button == 2) {
            mr = 0;
        }
    }, false);

    canvas.addEventListener("mousemove", (ev) => {
        const oEvent = ev || event;
        if (ml == 1) { // Rotate
            ang1 += (oEvent.clientX - mx) * 0.002;
            ang2 += (oEvent.clientY - my) * 0.002;
            if (oEvent.clientX != mx || oEvent.clientY != my) mm = 1;
        }
        if (mr == 1) { // Pan
            const l = len * 4.0 / (cx + cy);
            cenx += l * (-(oEvent.clientX - mx) * Math.sin(ang1) - (oEvent.clientY - my) * Math.sin(ang2) * Math.cos(ang1));
            ceny += l * ((oEvent.clientY - my) * Math.cos(ang2));
            cenz += l * ((oEvent.clientX - mx) * Math.cos(ang1) - (oEvent.clientY - my) * Math.sin(ang2) * Math.sin(ang1));
            if (oEvent.clientX != mx || oEvent.clientY != my) mm = 1;
        }
        mx = oEvent.clientX;
        my = oEvent.clientY;
    }, false);

    canvas.addEventListener("wheel", (ev) => { // Zoom
        ev.preventDefault();
        const oEvent = ev || event;
        len *= Math.exp(0.001 * oEvent.deltaY);
    }, false);

    // Prevent context menu on right-click drag
    document.oncontextmenu = (event) => {
        if (mm == 1) event.preventDefault();
    };

    // --- Touch Controls ---
    canvas.addEventListener("touchstart", (ev) => {
        ev.preventDefault();
        const n = ev.touches.length;
        if (n == 1) { // Rotate
            if (isRendering) rotationSpeed = 0; // Stop rotation
            const oEvent = ev.touches[0];
            mx = oEvent.clientX;
            my = oEvent.clientY;
        } else if (n == 2) { // Pan/Zoom
            const oEvent = ev.touches[0];
            const oEvent1 = ev.touches[1];
            mx = oEvent.clientX;
            my = oEvent.clientY;
            mx1 = oEvent1.clientX;
            my1 = oEvent1.clientY;
        }
        lasttimen = n;
    }, { passive: false });

    canvas.addEventListener("touchend", (ev) => {
        ev.preventDefault();
        if (isRendering && lasttimen == 1 && ev.touches.length == 0) {
            rotationSpeed = parseInt(speedSlider.value, 10); // Resume rotation
        }
        const n = ev.touches.length;
        if (n == 1) {
            const oEvent = ev.touches[0];
            mx = oEvent.clientX;
            my = oEvent.clientY;
        } else if (n == 2) {
            const oEvent = ev.touches[0];
            const oEvent1 = ev.touches[1];
            mx = oEvent.clientX;
            my = oEvent.clientY;
            mx1 = oEvent1.clientX;
            my1 = oEvent1.clientY;
        }
        lasttimen = n;
    }, { passive: false });

    canvas.addEventListener("touchmove", (ev) => {
        ev.preventDefault();
        const n = ev.touches.length;
        if (n == 1 && lasttimen == 1) { // Rotate
            const oEvent = ev.touches[0];
            ang1 += (oEvent.clientX - mx) * 0.002;
            ang2 += (oEvent.clientY - my) * 0.002;
            mx = oEvent.clientX;
            my = oEvent.clientY;
        } else if (n == 2) {
            const oEvent = ev.touches[0];
            const oEvent1 = ev.touches[1];
            
            // Pan
            const l = len * 2.0 / (cx + cy);
            cenx += l * (-(oEvent.clientX + oEvent1.clientX - mx - mx1) * Math.sin(ang1) - (oEvent.clientY + oEvent1.clientY - my - my1) * Math.sin(ang2) * Math.cos(ang1));
            ceny += l * ((oEvent.clientY + oEvent1.clientY - my - my1) * Math.cos(ang2));
            cenz += l * ((oEvent.clientX + oEvent1.clientX - mx - mx1) * Math.cos(ang1) - (oEvent.clientY + oEvent1.clientY - my - my1) * Math.sin(ang2) * Math.sin(ang1));
            
            // Zoom
            const l1 = Math.sqrt((mx - mx1) * (mx - mx1) + (my - my1) * (my - my1) + 1.0);
            mx = oEvent.clientX;
            my = oEvent.clientY;
            mx1 = oEvent1.clientX;
            my1 = oEvent1.clientY;
            const l2 = Math.sqrt((mx - mx1) * (mx - mx1) + (my - my1) * (my - my1) + 1.0);
            len *= l1 / l2;
        }
        lasttimen = n;
    }, { passive: false });
}

// --- App Entry Point ---

window.onload = function () {
    detectMobile();
    checkHWA();
    
    // Restore saved panel width
    const savedWidth = localStorage.getItem('panelWidth');
    if (savedWidth && !isMobile) {
        panel.style.width = savedWidth;
    }
    
    initGL();
    initMonaco();
    initEventListeners();
    initCanvasControls();
    updatePresetList();
    
    // Set initial render state on button
    if (isRendering) {
        renderText.textContent = "Pause Rendering";
        renderIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
    } else {
        renderText.textContent = "Start Rendering";
        renderIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
    }
};