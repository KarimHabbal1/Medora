import React, { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { triageApi } from '../../api/triage';
import Button from '../../components/ui/Button';
import Badge from '../../components/ui/Badge';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import type { TriageSession, Message } from '../../types/triage';
import { TriageSessionStatus, MessageSender } from '../../types/enums';

const TriageSessionPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [session, setSession] = useState<TriageSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [ending, setEnding] = useState(false);
  const [error, setError] = useState('');
  const [ended, setEnded] = useState(false);
  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sessionId) return;
    const load = async () => {
      try {
        const [s, m] = await Promise.all([
          triageApi.getSession(sessionId),
          triageApi.getMessages(sessionId),
        ]);
        setSession(s);
        setMessages(m);
        if (s.status !== TriageSessionStatus.Active) setEnded(true);
      } catch { setError('Failed to load session.'); }
      finally { setLoading(false); }
    };
    load();
  }, [sessionId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

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
      setMessages((prev) => {
        // Replace temp msg with server version if needed, add agent reply
        const withoutTemp = prev.filter((m) => m.id !== tempMsg.id);
        return [...withoutTemp, { ...tempMsg, id: `patient-${Date.now()}` }, agentReply];
      });
    } catch {
      setError('Failed to send message. Please try again.');
    } finally {
      setSending(false);
    }
  };

  const handleEnd = async () => {
    if (!sessionId) return;
    setEnding(true);
    setError('');
    try {
      await triageApi.endSession(sessionId);
      setEnded(true);
      setShowEndConfirm(false);
      const updatedSession = await triageApi.getSession(sessionId);
      setSession(updatedSession);
    } catch { setError('Failed to end session.'); }
    finally { setEnding(false); }
  };

  const isAgent = (sender: MessageSender) =>
    sender !== MessageSender.Patient;

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto flex flex-col h-[calc(100vh-7rem)] animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div>
          <h1 className="text-lg font-semibold text-text-primary">
            {session?.chief_complaint || 'Triage Session'}
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
        {!ended && (
          <Button variant="danger" size="sm" onClick={() => setShowEndConfirm(true)}>
            End Session
          </Button>
        )}
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} className="mb-3 flex-shrink-0" />}

      {/* End confirm modal */}
      {showEndConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <div className="bg-white rounded-xl border border-border shadow-modal p-6 max-w-sm mx-4">
            <h3 className="text-lg font-semibold text-text-primary mb-2">End Session?</h3>
            <p className="text-sm text-text-secondary mb-4">
              This will finalize the triage and generate a report for your doctor. You won&apos;t be able to send more messages.
            </p>
            <div className="flex justify-end gap-2">
              <Button variant="secondary" onClick={() => setShowEndConfirm(false)}>Cancel</Button>
              <Button variant="danger" onClick={handleEnd} loading={ending}>End Session</Button>
            </div>
          </div>
        </div>
      )}

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto bg-white rounded-xl border border-border p-4 chat-scroll">
        {messages.length === 0 && !ended && (
          <div className="flex items-center justify-center h-full text-text-tertiary text-sm">
            Start by describing your symptoms...
          </div>
        )}
        <div className="space-y-3">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${isAgent(msg.sender) ? 'justify-start' : 'justify-end'}`}>
              <div
                className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                  isAgent(msg.sender)
                    ? 'bg-surface-tertiary text-text-primary rounded-bl-md'
                    : 'bg-medora-600 text-white rounded-br-md'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
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
