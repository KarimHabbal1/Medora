import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/ui/Button';

const UnauthorizedPage: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-secondary px-4">
      <div className="text-center">
        <div className="text-6xl font-bold text-medora-200 mb-4">403</div>
        <h1 className="text-2xl font-semibold text-text-primary mb-2">Access Denied</h1>
        <p className="text-text-secondary mb-8 max-w-md">
          You don&apos;t have permission to access this page. Please contact your administrator if you believe this is an error.
        </p>
        <Link to="/">
          <Button>Go Home</Button>
        </Link>
      </div>
    </div>
  );
};

export default UnauthorizedPage;
