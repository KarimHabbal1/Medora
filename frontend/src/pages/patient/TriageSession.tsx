import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { triageApi } from '../../api/triage';
import Button from '../../components/ui/Button';
import Badge from '../../components/ui/Badge';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import type { PatientTriageSession, Message, SessionPhase } from '../../types/triage';
import { TriageSessionStatus, MessageSender, MessageType, AgentPhase } from '../../types/enums';

/**
 * Phase indicator labels — maps agent phases to patient-friendly text.
 * No diagnosis data is ever shown to the patient.
 */
const PHASE_LABELS: Record<string, { label: string; icon: string; color: string }> = {
  [AgentPhase.Intake]: {
    label: 'Describing your symptoms',
    icon: '💬',
    color: 'bg-blue-50 text-blue-700 border-blue-200',
  },
  [AgentPhase.TriageModeA]: {
    label: 'Reviewing your information',
    icon: '🔍',
    color: 'bg-indigo-50 text-indigo-700 border-indigo-200',
  },
  [AgentPhase.TriageModeB]: {
    label: 'Follow-up questions',
    icon: '🩺',
    color: 'bg-purple-50 text-purple-700 border-purple-200',
  },
  [AgentPhase.Escalated]: {
    label: 'Urgent attention required',
    icon: '🚨',
    color: 'bg-red-50 text-red-700 border-red-200',
  },
  [AgentPhase.Completed]: {
    label: 'Assessment complete',
    icon: '✅',
    color: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  },
};

const TriageSessionPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [session, setSession] = useState<PatientTriageSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [phase, setPhase] = useState<SessionPhase | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [, setEnding] = useState(false);
  const [error, setError] = useState('');
  const [ended, setEnded] = useState(false);
  const [, setShowEndConfirm] = useState(false);
  const [, setStreamingId] = useState<string | null>(null);
  const [, setStreamedText] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load session data
  useEffect(() => {
    if (!sessionId) return;
    const load = async () => {
      try {
        const [s, m, p] = await Promise.all([
          triageApi.getSession(sessionId),
          triageApi.getMessages(sessionId),
          triageApi.getSessionPhase(sessionId),
        ]);
        setSession(s);
        setMessages(m);
        setPhase(p);
        if (s.status !== TriageSessionStatus.Active) setEnded(true);
      } catch { setError('Failed to load session.'); }
      finally { setLoading(false); }
    };
    load();
  }, [sessionId]);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  // Refresh phase after each agent response
  const refreshPhase = useCallback(async () => {
    if (!sessionId) return;
    try {
      const p = await triageApi.getSessionPhase(sessionId);
      setPhase(p);
      if (p.phase === AgentPhase.Completed || p.phase === AgentPhase.Escalated) {
        const updatedSession = await triageApi.getSession(sessionId);
        setSession(updatedSession);
        if (updatedSession.status !== TriageSessionStatus.Active) {
          setEnded(true);
        }
      }
    } catch { /* phase refresh is best-effort */ }
  }, [sessionId]);

  const handleSend = async () => {
    if (!input.trim() || !sessionId || sending) return;
    const text = input.trim();
    setInput('');
    setSending(true);
    setError('');

    // Optimistic patient message
    const tempMsg: Message = {
      id: `temp-${Date.now()}`,
      session_id: sessionId,
      sender: MessageSender.Patient,
      content: text,
      message_type: 'text' as Message['message_type'],
      is_persisted_after_summary: true,
      is_visible_to_doctor: true,
      is_deleted: false,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempMsg]);

    try {
      const agentReply = await triageApi.sendMessage(sessionId, { content: text });
      const replyId = agentReply.id;
      // Add reply with empty content — we'll stream it in
      setMessages((prev) => {
        const withoutTemp = prev.filter((m) => m.id !== tempMsg.id);
        return [...withoutTemp, { ...tempMsg, id: `patient-${Date.now()}` }, { ...agentReply, content: '' }];
      });
      setSending(false);

      // Stream words in
      const words = agentReply.content.split(' ');
      setStreamingId(replyId);
      setStreamedText('');
      for (let i = 0; i < words.length; i++) {
        const partial = words.slice(0, i + 1).join(' ');
        setStreamedText(partial);
        setMessages((prev) => prev.map((m) => m.id === replyId ? { ...m, content: partial } : m));
        await new Promise((r) => setTimeout(r, 30));
      }
      setStreamingId(null);
      setStreamedText('');

      await refreshPhase();
    } catch {
      setError('Failed to send message. Please try again.');
      setSending(false);
    } finally {
      inputRef.current?.focus();
    }
  };

  const isAgent = (sender: MessageSender) =>
    sender !== MessageSender.Patient;

  const isEscalation = (msg: Message) =>
    msg.message_type === MessageType.Escalation || msg.message_type === ('escalation' as MessageType);

  const currentPhaseInfo = phase?.phase ? PHASE_LABELS[phase.phase] : null;

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto flex flex-col h-[calc(100vh-7rem)] animate-fade-in">
      {/* Back navigation */}
      <button
        onClick={() => navigate(-1)}
        className="self-start flex items-center gap-1 text-sm text-text-secondary hover:text-text-primary mb-3 transition-colors"
      >
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
        </svg>
        Back
      </button>

      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div>
          <h1 className="text-lg font-semibold text-text-primary">
            Triage Session
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant={ended ? 'success' : 'primary'}>
              {session?.status || 'unknown'}
            </Badge>
            <span className="text-xs text-text-tertiary">
              {session && new Date(session.started_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>

      {/* Phase indicator */}
      {currentPhaseInfo && (
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm mb-3 flex-shrink-0 transition-all duration-300 ${currentPhaseInfo.color}`}>
          <span className="text-base">{currentPhaseInfo.icon}</span>
          <span className="font-medium">{currentPhaseInfo.label}</span>
        </div>
      )}

      {/* Escalation banner — safety-critical, always shown to patient */}
      {phase?.is_escalated && (
        <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 mb-3 flex-shrink-0 animate-pulse">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl">🚨</span>
            <h3 className="text-base font-bold text-red-800">Urgent Attention Required</h3>
          </div>
          <p className="text-sm text-red-700">
            Based on your symptoms, you should seek immediate medical attention.
            Please follow the instructions provided in the chat below.
          </p>
        </div>
      )}

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} className="mb-3 flex-shrink-0" />}

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto bg-white rounded-xl border border-border p-4 chat-scroll">
        {messages.length === 0 && !ended && (
          <div className="flex items-center justify-center h-full text-text-tertiary text-sm">
            Start by describing your symptoms...
          </div>
        )}
        <div className="space-y-3">
          {messages
            .filter((msg) => {
              if (msg.sender === MessageSender.System && msg.content.includes('CLINICIAN HANDOVER')) return false;
              return true;
            })
            .map((msg) => {
            const isEscalationMsg = isEscalation(msg);
            const displayContent = isEscalationMsg
              ? 'Based on your symptoms, please seek immediate medical attention. Your report has been sent to your doctor for urgent review.'
              : msg.content;

            return (
            <div key={msg.id} className={`flex ${isAgent(msg.sender) ? 'justify-start' : 'justify-end'}`}>
              <div
                className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                  isEscalationMsg
                    ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-md'
                    : isAgent(msg.sender)
                    ? 'bg-surface-tertiary text-text-primary rounded-bl-md'
                    : 'bg-medora-600 text-white rounded-br-md'
                }`}
              >
                {isAgent(msg.sender) && (
                  <span className="text-[10px] font-semibold uppercase tracking-wider opacity-60 block mb-1">
                    {msg.sender === MessageSender.IntakeAgent
                      ? 'Medora Intake'
                      : msg.sender === MessageSender.TriageAgent
                      ? 'Medora Clinical'
                      : msg.sender === MessageSender.System
                      ? 'System'
                      : 'Medora'}
                  </span>
                )}
                {displayContent}
              </div>
            </div>
            );
          })}
          {sending && (
            <div className="flex justify-start">
              <div className="bg-surface-tertiary px-4 py-3 rounded-2xl rounded-bl-md">
                <div className="flex gap-1.5">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              </div>
            </div>
          )}
          {ended && (
            <div className="flex justify-center my-4">
              <div className="bg-emerald-50 border border-emerald-200 text-emerald-700 text-sm px-4 py-3 rounded-lg text-center max-w-sm">
                <svg className="h-5 w-5 mx-auto mb-1 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Your report has been sent to your doctor.
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>

      {/* Input bar */}
      {!ended && (
        <div className="flex gap-2 mt-3 flex-shrink-0">
          <input
            ref={inputRef}
            autoFocus
            className="flex-1 rounded-xl border border-border bg-white px-4 py-2.5 text-sm text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-2 focus:ring-medora-500 focus:border-medora-500 hover:border-medora-300 transition-colors"
            placeholder="Describe your symptoms..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            disabled={sending}
          />
          <Button onClick={handleSend} disabled={!input.trim() || sending}>
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </Button>
        </div>
      )}
    </div>
  );
};

export default TriageSessionPage;
