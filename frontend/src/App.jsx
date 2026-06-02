import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Send, Loader2 } from 'lucide-react';

const API_URL = "http://localhost:8000";
const cn = (...classes) => classes.filter(Boolean).join(' ');

export default function App() {
  const [step, setStep] = useState('splash');
  const [messages, setMessages] = useState([{ role: 'ai', text: "Welcome. I am O.P. Jindal." }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const mediaRef = useRef(null);
  const chunks = useRef([]);
  const endRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    if (step === 'splash') setTimeout(() => setStep('main'), 3000);
  }, [step]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Play Edge TTS audio
  const playAudio = (audioData) => {
    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      const src = `data:audio/mpeg;base64,${audioData}`;
      const audio = new Audio(src);
      audioRef.current = audio;

      audio.onplay = () => setIsSpeaking(true);
      audio.onended = () => {
        setIsSpeaking(false);
        audioRef.current = null;
      };
      audio.onerror = (e) => {
        console.error("Audio error:", e);
        setIsSpeaking(false);
        audioRef.current = null;
      };

      audio.play().catch(err => {
        console.error("Audio play failed:", err);
        setIsSpeaking(false);
        audioRef.current = null;
      });
    } catch (e) {
      console.error("Audio setup error:", e);
      setIsSpeaking(false);
    }
  };

  const processResponse = async (res) => {
    try {
      const data = await res.json();
      console.log("Server response:", data);

      const aiText = data.text ?? data.response ?? data.answer ?? data.reply ?? "";
      setMessages(prev => [...prev, { role: 'ai', text: aiText || "No answer received." }]);

      // Play audio if available
      if (data.audio_data) {
        playAudio(data.audio_data);
      }

      // Play video if available (D-ID)
      if (data.video_url) {
        setVideoUrl(data.video_url);
      }
    } catch (e) {
      console.error("Error parsing response", e);
    }
    setLoading(false);
  };

  const handleSendText = async () => {
    if (!input.trim() || loading) return;

    const txt = input;
    setMessages(p => [...p, { role: 'user', text: txt }]);
    setLoading(true);
    setInput('');

    try {
      const res = await fetch(`${API_URL}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: txt })
      });
      await processResponse(res);
    } catch {
      setLoading(false);
    }
  };

  const startVoice = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRef.current = new MediaRecorder(s);
      chunks.current = [];

      mediaRef.current.ondataavailable = e => chunks.current.push(e.data);

      mediaRef.current.onstop = async () => {
        setLoading(true);

        const fd = new FormData();
        fd.append("file", new Blob(chunks.current, { type: "audio/webm" }), "in.webm");

        try {
          const res = await fetch(`${API_URL}/voice_chat/`, { method: "POST", body: fd });
          await processResponse(res);
        } catch (err) {
          console.error(err);
          setLoading(false);
        }
      };

      mediaRef.current.start();
      setIsRecording(true);
    } catch {
      alert("Mic blocked");
    }
  };

  return (
    <div className="h-screen w-full bg-[#FAF9F6] text-stone-900 flex items-center justify-center font-sans overflow-hidden relative">
      <AnimatePresence mode="wait">
        {step === 'splash' ? (
          <motion.div key="s" exit={{ opacity: 0 }} className="text-center">
            <h1 className="text-7xl font-serif tracking-tighter">O.P. JINDAL</h1>
            <p className="text-stone-400 tracking-[0.6em] mt-4 uppercase font-bold">Digital Legacy</p>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="w-full h-full flex flex-col p-8 lg:p-12 gap-8"
          >
            <div className="flex-1 flex flex-col lg:flex-row gap-12 overflow-hidden">
              {/* LEFT: LOG */}
              <aside className="hidden lg:flex flex-col w-[350px] border-r border-stone-200 pr-12 overflow-y-auto no-scrollbar">
                <p className="text-[10px] font-bold uppercase tracking-widest text-stone-400 mb-10">Chronology</p>
                <div className="space-y-8">
                  {messages.map((m, i) => (
                    <div key={i} className="space-y-1">
                      <p className="text-[9px] font-bold text-stone-300 uppercase">{m.role}</p>
                      <p className="text-xs leading-relaxed text-stone-600">{m.text}</p>
                    </div>
                  ))}
                  {loading && <div className="text-[10px] animate-pulse text-stone-400 uppercase">Generating...</div>}
                  <div ref={endRef} />
                </div>
              </aside>

              {/* CENTER: PORTRAIT */}
              <main className="flex-1 flex flex-col items-center justify-center relative">
                <div
                  className={cn(
                    "w-80 h-80 lg:w-[500px] lg:h-[500px] rounded-full border border-stone-200 p-4 transition-all duration-1000 bg-white shadow-2xl relative overflow-hidden",
                    isSpeaking ? "scale-105 border-stone-400" : "grayscale-[0.8] opacity-80"
                  )}
                >
                  {videoUrl ? (
                    <video
                      src={videoUrl}
                      autoPlay
                      onEnded={() => setVideoUrl(null)}
                      className="w-full h-full object-cover rounded-full scale-[1.05]"
                    />
                  ) : (
                    <img
                      src="/op.png"
                      className="w-full h-full object-cover rounded-full"
                      alt="Portrait"
                    />
                  )}
                </div>

                <div className="mt-12 text-center">
                  <h2 className="text-4xl font-serif tracking-tight">Om Prakash Jindal</h2>
                  <p className="text-stone-400 text-[10px] uppercase tracking-[0.4em] mt-2 font-bold">
                    {isSpeaking ? 'Speaking' : 'Ready'}
                  </p>
                </div>
              </main>
            </div>

            {/* INPUT */}
            <footer className="max-w-3xl w-full mx-auto pb-10">
              <div className="flex items-center bg-white border border-stone-200 p-2 rounded-full shadow-sm focus-within:border-stone-900 transition-all">
                <button
                  onClick={() => { if (isRecording) { mediaRef.current?.stop(); setIsRecording(false); } else { startVoice(); } }}
                  className={cn(
                    "w-14 h-14 rounded-full flex items-center justify-center transition-all",
                    isRecording ? "bg-red-500 text-white" : "bg-stone-50 text-stone-900"
                  )}
                >
                  {loading ? <Loader2 className="animate-spin text-stone-400" size={20} /> : <Mic size={20} />}
                </button>

                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Inquire about the journey..."
                  className="flex-1 bg-transparent border-none outline-none px-6 text-sm font-light"
                  onKeyDown={(e) => e.key === 'Enter' && handleSendText()}
                  disabled={loading}
                />

                <button onClick={handleSendText} disabled={loading} className="text-stone-400 hover:text-stone-900 mr-4 transition-colors">
                  <Send size={20} />
                </button>
              </div>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}