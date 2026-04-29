import React from 'react';
import Card from '../ui/Card';

interface ReportSectionCardProps {
  title: string;
  content: unknown;
  variant?: 'default' | 'warning';
}

function renderContent(content: unknown): React.ReactNode {
  if (content === null || content === undefined) return <span className="text-text-tertiary italic">N/A</span>;
  if (typeof content === 'string') return <p className="text-sm text-text-primary whitespace-pre-wrap">{content}</p>;
  if (Array.isArray(content)) {
    if (content.length === 0) return <span className="text-text-tertiary italic">None</span>;
    return (
      <ul className="list-disc list-inside space-y-1">
        {content.map((item, i) => (
          <li key={i} className="text-sm text-text-primary">{typeof item === 'string' ? item : JSON.stringify(item)}</li>
        ))}
      </ul>
    );
  }
  if (typeof content === 'object') {
    const obj = content as Record<string, unknown>;
    if (obj.items && Array.isArray(obj.items)) return renderContent(obj.items);
    return <pre className="text-sm text-text-primary whitespace-pre-wrap">{JSON.stringify(content, null, 2)}</pre>;
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
