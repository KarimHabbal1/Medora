import React from 'react';
import Card from '../ui/Card';

interface ReportSectionCardProps {
  title: string;
  content: unknown;
  variant?: 'default' | 'warning';
}

function formatKey(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderContent(content: unknown): React.ReactNode {
  if (content === null || content === undefined) return <span className="text-text-tertiary italic">N/A</span>;
  if (typeof content === 'string') return <p className="text-sm text-text-primary whitespace-pre-wrap">{content}</p>;
  if (typeof content === 'number' || typeof content === 'boolean') return <p className="text-sm text-text-primary">{String(content)}</p>;
  if (Array.isArray(content)) {
    if (content.length === 0) return <span className="text-text-tertiary italic">None</span>;
    return (
      <ul className="list-disc list-inside space-y-1">
        {content.map((item, i) => (
          <li key={i} className="text-sm text-text-primary">
            {typeof item === 'string' ? item
              : typeof item === 'object' && item !== null
                ? Object.entries(item as Record<string, unknown>).map(([k, v]) => `${formatKey(k)}: ${String(v)}`).join(' — ')
                : String(item)}
          </li>
        ))}
      </ul>
    );
  }
  if (typeof content === 'object') {
    const obj = content as Record<string, unknown>;
    if (obj.items && Array.isArray(obj.items)) return renderContent(obj.items);
    const entries = Object.entries(obj).filter(([, v]) => v !== null && v !== undefined);
    if (entries.length === 0) return <span className="text-text-tertiary italic">N/A</span>;
    return (
      <dl className="space-y-2">
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt className="text-xs font-semibold text-text-secondary">{formatKey(key)}</dt>
            <dd className="text-sm text-text-primary ml-2">{typeof value === 'object' ? renderContent(value) : String(value)}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return <p className="text-sm text-text-primary">{String(content)}</p>;
}

const ReportSectionCard: React.FC<ReportSectionCardProps> = ({ title, content, variant = 'default' }) => {
  return (
    <Card className={variant === 'warning' ? 'border-amber-200 bg-amber-50/50' : ''} padding="sm">
      <h3 className={`text-sm font-semibold mb-2 ${variant === 'warning' ? 'text-amber-800' : 'text-text-secondary'}`}>{title}</h3>
      {renderContent(content)}
    </Card>
  );
};

export default ReportSectionCard;
