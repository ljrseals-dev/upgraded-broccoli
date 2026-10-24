"""Local browser UI: a tiny HTTP server (stdlib) bound to 127.0.0.1 that serves one page and the JSON protocol.
Protection: random per-run token (header or query), Host and Origin checks (blocks DNS rebinding and cross-site
requests), 1 MB request cap, files served only from inside the workspace."""
from __future__ import annotations
import json, mimetypes, secrets, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from . import __version__, EDITION
from .protocol import Engine

MAX_BODY = 1_000_000

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AOC Magic Studio</title><style>
:root{--bg:#070b14;--panel:#0d1424;--line:#1c2a44;--txt:#dce6ff;--dim:#8ea0c4;--cy:#38d6ff;--am:#ffb347;--gr:#35e0a1;--rd:#ff5c7a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 500px at 50% -10%,#10305a55,transparent),var(--bg);color:var(--txt);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px}
h1{margin:0;font-size:22px;letter-spacing:.18em;font-weight:600}h1 b{color:var(--cy)}h1 i{color:var(--am);font-style:normal}
.tag{color:var(--dim);font-size:12px;letter-spacing:.12em}nav{display:flex;gap:6px;padding:10px 24px;border-bottom:1px solid var(--line)}
nav button{background:none;border:1px solid var(--line);color:var(--dim);padding:7px 14px;border-radius:8px;cursor:pointer}
nav button.on{color:var(--txt);border-color:var(--cy);box-shadow:0 0 0 1px #38d6ff33 inset}
main{padding:20px 24px;max-width:1100px;margin:auto}section{display:none}section.on{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
label{display:block;color:var(--dim);font-size:12px;margin:8px 0 3px}textarea,input,select{width:100%;background:#09101d;border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:8px}
textarea{min-height:90px;font-family:ui-monospace,Consolas,monospace}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
button.go{background:linear-gradient(90deg,#1394c7,#2b6ff0);border:0;color:#fff;padding:10px 18px;border-radius:9px;cursor:pointer;margin-top:12px;font-weight:600}
button.go:disabled{opacity:.5}.msg{margin-top:10px;color:var(--dim);white-space:pre-wrap}.err{color:var(--rd)}.ok{color:var(--gr)}
canvas{width:100%;height:90px;background:#09101d;border:1px solid var(--line);border-radius:8px;margin-top:10px}audio{width:100%;margin-top:10px}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}th{color:var(--dim);font-weight:500}
td input[type=range]{padding:0}.pill{font-size:11px;color:var(--am);border:1px solid #ffb34755;border-radius:20px;padding:1px 8px}
</style></head><body>
<header><h1><b>AOC</b> MAGIC <i>STUDIO</i></h1><span class="tag">STANDALONE EDITION · v__VER__ · OFFLINE · NUMPY ONLY</span></header>
<nav><button data-t="speak" class="on">Speak</button><button data-t="studio">Vocal Studio</button><button data-t="status">Status</button></nav>
<main>
<section id="speak" class="on"><div class="card"><label>Text</label><textarea id="sp_text">Hello, this is AOC Magic Studio.</textarea>
<div class="row"><div><label>Engine</label><select id="sp_eng"><option>auto</option><option>builtin</option><option>espeak</option></select></div>
<div><label>Rate</label><input id="sp_rate" type="number" min="0.4" max="2.5" step="0.1" value="1.0"></div>
<div><label>Pitch (Hz)</label><input id="sp_pitch" type="number" min="60" max="300" value="118"></div>
<div><label>AOC character chain</label><select id="sp_char"><option value="0">off</option><option value="1">on (+2 st, 1.15x)</option></select></div></div>
<button class="go" id="sp_go">Synthesize</button><div class="msg" id="sp_msg"></div><audio id="sp_audio" controls hidden></audio><canvas id="sp_wave"></canvas></div></section>
<section id="studio"><div class="card"><label>Lead melody (note:beats, R = rest; e.g. C4:1 D4:1 E4:2). Beats use the tempo below.</label>
<textarea id="st_lead">C4:1 D4:1 E4:1 G4:1 F4:1 E4:1 D4:1 C4:2</textarea>
<div class="row"><div><label>Key</label><select id="st_key"><option>C</option><option>C#</option><option>D</option><option>Eb</option><option>E</option><option>F</option><option>F#</option><option>G</option><option>Ab</option><option>A</option><option>Bb</option><option>B</option></select></div>
<div><label>Mode</label><select id="st_mode"><option>major</option><option>minor</option></select></div>
<div><label>Parts</label><select id="st_parts"><option>2</option><option>3</option><option selected>4</option><option>6</option></select></div>
<div><label>Doubles</label><input id="st_dbl" type="number" min="0" max="8" value="1"></div>
<div><label>Backgrounds</label><select id="st_bg"><option>airy</option><option>soulful</option><option>rhythmic</option><option>powerful</option><option>none</option></select></div>
<div><label>Tempo (BPM)</label><input id="st_tempo" type="number" min="20" max="300" value="100"></div></div>
<button class="go" id="st_build">Build arrangement</button> <button class="go" id="st_render" disabled>Render stems + mix</button> <button class="go" id="st_midi" disabled>Export MIDI</button>
<div class="msg" id="st_msg"></div></div>
<div class="card" id="st_tracks" hidden><span class="pill">preview formant synth: not a realistic or cloned voice</span><table><thead><tr><th>Track</th><th>Mute</th><th>Solo</th><th>Level dB</th><th>Pan</th><th>Reverb</th></tr></thead><tbody id="st_body"></tbody></table>
<audio id="st_audio" controls hidden></audio><div class="msg" id="st_links"></div></div></section>
<section id="status"><div class="card"><button class="go" id="do_go">Run doctor</button><div class="msg" id="do_out"></div></div></section>
</main><script>
const TOKEN="__TOKEN__";const $=id=>document.getElementById(id);
async function api(req){const r=await fetch("/api",{method:"POST",headers:{"Content-Type":"application/json","X-AOC-Token":TOKEN},body:JSON.stringify(req)});return r.json();}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button,section").forEach(e=>e.classList.remove("on"));b.classList.add("on");$(b.dataset.t).classList.add("on");});
const msg=(el,t,c)=>{el.className="msg "+(c||"");el.textContent=t;};
const fileUrl=f=>"/files/"+f.split("/").map(encodeURIComponent).join("/")+"?token="+TOKEN+"&t="+Date.now();
async function wave(url,cv){try{const ab=await (await fetch(url)).arrayBuffer();const ctx=new (window.AudioContext||window.webkitAudioContext)();const buf=await ctx.decodeAudioData(ab);const d=buf.getChannelData(0);
const w=cv.width=cv.clientWidth*2,h=cv.height=cv.clientHeight*2,g=cv.getContext("2d");g.clearRect(0,0,w,h);g.strokeStyle="#38d6ff";g.beginPath();const step=Math.max(1,Math.floor(d.length/w));
for(let x=0;x<w;x++){let mn=1,mx=-1;for(let i=0;i<step;i++){const v=d[x*step+i]||0;if(v<mn)mn=v;if(v>mx)mx=v;}g.moveTo(x,(1-mx)*h/2);g.lineTo(x,(1-mn)*h/2);}g.stroke();ctx.close();}catch(e){}}
$("sp_go").onclick=async()=>{const b=$("sp_go");b.disabled=true;msg($("sp_msg"),"Synthesizing...");
const r=await api({op:"speech.synth",text:$("sp_text").value,engine:$("sp_eng").value,rate:+$("sp_rate").value,pitch_hz:+$("sp_pitch").value,character:$("sp_char").value==="1",out:"ui/speech.wav"});b.disabled=false;
if(!r.ok)return msg($("sp_msg"),r.error.code+": "+r.error.message,"err");const u=fileUrl(r.result.file);const a=$("sp_audio");a.src=u;a.hidden=false;a.play().catch(()=>{});
msg($("sp_msg"),"engine: "+r.result.engine+" · "+r.result.duration_s+" s · "+r.result.sr+" Hz","ok");wave(u,$("sp_wave"));};
const PC={C:0,D:2,E:4,F:5,G:7,A:9,B:11};
function parseLead(txt,tempo){let t=0,out=[];const beat=60/tempo;for(const tok of txt.trim().split(/\s+/)){if(!tok)continue;const m=tok.match(/^([A-Ga-gR])([#b]?)(-?\d*):([\d.]+)$/);if(!m)throw new Error("bad token: "+tok);
const d=parseFloat(m[4])*beat;if(m[1].toUpperCase()==="R"){t+=d;continue;}let p=PC[m[1].toUpperCase()]+(m[2]==="#"?1:m[2]==="b"?-1:0);const oct=m[3]===""?4:parseInt(m[3]);out.push([12*(oct+1)+p,+t.toFixed(4),+d.toFixed(4),0.8,"la"]);t+=d;}
if(!out.length)throw new Error("no notes");return out;}
let TR=[];
function drawTracks(){const tb=$("st_body");tb.innerHTML="";TR.forEach(t=>{const tr=document.createElement("tr");const s=t.settings;
tr.innerHTML=`<td>${t.name}<br><small style="color:var(--dim)">${t.role} · ${t.note_count} notes</small></td><td><input type="checkbox" data-k="mute" ${s.mute?"checked":""}></td><td><input type="checkbox" data-k="solo" ${s.solo?"checked":""}></td>
<td><input type="range" min="-24" max="12" step="0.5" value="${s.level_db}" data-k="level_db"></td><td><input type="range" min="-1" max="1" step="0.05" value="${s.pan}" data-k="pan"></td><td><input type="range" min="0" max="1" step="0.05" value="${s.reverb}" data-k="reverb"></td>`;
tr.querySelectorAll("input").forEach(i=>i.onchange=async()=>{const v=i.type==="checkbox"?i.checked:+i.value;const r=await api({op:"studio.set_track",name:t.name,set:{[i.dataset.k]:v}});if(!r.ok)msg($("st_msg"),r.error.message,"err");});tb.appendChild(tr);});$("st_tracks").hidden=false;}
$("st_build").onclick=async()=>{try{const tempo=+$("st_tempo").value;const lead=parseLead($("st_lead").value,tempo);
const r=await api({op:"studio.build",lead,tempo,key:$("st_key").value,mode:$("st_mode").value,parts:+$("st_parts").value,doubles:+$("st_dbl").value,bg:$("st_bg").value});
if(!r.ok)return msg($("st_msg"),r.error.code+": "+r.error.message,"err");TR=r.result.tracks;drawTracks();$("st_render").disabled=false;$("st_midi").disabled=false;msg($("st_msg"),TR.length+" tracks built","ok");}catch(e){msg($("st_msg"),e.message,"err");}};
$("st_render").onclick=async()=>{const b=$("st_render");b.disabled=true;msg($("st_msg"),"Rendering...");const r=await api({op:"studio.render",out_dir:"ui/render",bits:16});b.disabled=false;
if(!r.ok)return msg($("st_msg"),r.error.code+": "+r.error.message,"err");const a=$("st_audio");a.src=fileUrl(r.result.files.mix);a.hidden=false;
$("st_links").innerHTML=Object.entries(r.result.files).map(([k,f])=>`<a style="color:var(--cy)" href="${fileUrl(f)}" download>${k}</a>`).join(" · ");msg($("st_msg"),"rendered "+r.result.duration_s+" s with: "+r.result.renderer,"ok");};
$("st_midi").onclick=async()=>{const r=await api({op:"midi.export",out:"ui/arrangement.mid"});if(!r.ok)return msg($("st_msg"),r.error.message,"err");
$("st_links").innerHTML+=` · <a style="color:var(--cy)" href="${fileUrl(r.result.file)}" download>arrangement.mid</a>`;};
$("do_go").onclick=async()=>{const r=await api({op:"doctor"});$("do_out").textContent=r.ok?r.result.checks.map(c=>`[${c.level}] ${c.check}: ${c.detail}`+(c.level!=="PASS"&&c.fix?`\n       fix: ${c.fix}`:"")).join("\n"):r.error.message;};
</script></body></html>"""


def make_server(workspace=".", host="127.0.0.1", port=0, token=None):
    token = token or secrets.token_urlsafe(24)
    eng, lock = Engine(workspace), threading.Lock()

    class H(BaseHTTPRequestHandler):
        server_version = "AOC"
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _ok_host(self):
            host_h = (self.headers.get("Host") or "").lower()
            port_ = self.server.server_address[1]
            if host_h not in (f"127.0.0.1:{port_}", f"localhost:{port_}"):
                return False
            org = self.headers.get("Origin")
            return org is None or org in (f"http://127.0.0.1:{port_}", f"http://localhost:{port_}")

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):          # client went away; nothing to report
                self.close_connection = True

        def _err(self, code, msg, close=False):
            self._send(code, json.dumps({"ok": False, "error": {"code": "http_%d" % code, "message": msg}}).encode(),
                       extra={"Connection": "close"} if close else None)

        def _token_ok(self, q):
            given = self.headers.get("X-AOC-Token") or (q.get("token") or [""])[0]
            return secrets.compare_digest(given, token)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._ok_host():
                return self._err(403, "bad host or origin")
            if not self._token_ok(q):
                return self._err(403, "missing or bad token")
            if u.path == "/":
                return self._send(200, PAGE.replace("__TOKEN__", token).replace("__VER__", __version__).encode(), "text/html; charset=utf-8")
            if u.path.startswith("/files/"):
                try:
                    p = eng.path(unquote(u.path[len("/files/"):]), must_exist=True,
                                 suffixes=(".wav", ".mid", ".midi", ".json"))
                except Exception:
                    return self._err(404, "not found")
                if not p.is_file():
                    return self._err(404, "not found")
                ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                return self._send(200, p.read_bytes(), ctype)
            return self._err(404, "not found")

        def do_POST(self):
            u = urlparse(self.path)
            if not self._ok_host():
                return self._err(403, "bad host or origin")
            if not self._token_ok({}):
                return self._err(403, "missing or bad token")
            if u.path != "/api":
                return self._err(404, "not found")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._err(400, "bad length")
            if n <= 0 or n > MAX_BODY:
                self.close_connection = True                     # do not try to read an oversized/unknown body
                return self._err(413, "body must be 1..1000000 bytes", close=True)
            try:
                req = json.loads(self.rfile.read(n))
            except ValueError:
                return self._err(400, "invalid JSON")
            with lock:
                resp = eng.handle(req)
            self._send(200, json.dumps(resp).encode())

    srv = ThreadingHTTPServer((host, port), H)
    srv.aoc_token, srv.aoc_engine = token, eng
    return srv


def run(workspace=".", port=8765, open_browser=True):
    srv = make_server(workspace, port=port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?token={srv.aoc_token}"
    print(f"AOC Magic Studio {__version__} {EDITION}\nworkspace: {srv.aoc_engine.ws}\nopen: {url}\n(Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
