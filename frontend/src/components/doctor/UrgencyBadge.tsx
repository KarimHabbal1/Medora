import React from 'react';
import Badge from '../ui/Badge';

interface UrgencyBadgeProps {
  level: string;
  size?: 'sm' | 'md';
}

const UrgencyBadge: React.FC<UrgencyBadgeProps> = ({ level, size = 'sm' }) => {
  const map: Record<string, { variant: 'success' | 'warning' | 'danger' | 'default'; label: string }> = {
    routine: { variant: 'success', label: 'Routine' },
    urgent: { variant: 'warning', label: 'Urgent' },
    emergency: { variant: 'danger', label: 'Emergency' },
    unknown: { variant: 'default', label: 'Unknown' },
  };
  const info = map[level] || map.unknown;
  return <Badge variant={info.variant} size={size}>{info.label}</Badge>;
};

export default UrgencyBadge;
