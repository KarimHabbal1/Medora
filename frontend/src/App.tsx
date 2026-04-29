import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { useAuth } from './hooks/useAuth';
import { UserRole } from './types/enums';

// Layouts
import AppLayout from './components/layout/AppLayout';
import AuthLayout from './components/layout/AuthLayout';

// Guards
import ProtectedRoute from './components/auth/ProtectedRoute';
import RoleGuard from './components/auth/RoleGuard';

// Auth pages
import LoginPage from './pages/auth/LoginPage';
import SignupPage from './pages/auth/SignupPage';

// Patient pages
import PatientDashboard from './pages/patient/PatientDashboard';
import PatientProfile from './pages/patient/PatientProfile';
import MedicalHistory from './pages/patient/MedicalHistory';
import TriageList from './pages/patient/TriageList';
import TriageSession from './pages/patient/TriageSession';

// Doctor pages
import DoctorDashboard from './pages/doctor/DoctorDashboard';
import DoctorPatients from './pages/doctor/DoctorPatients';
import DoctorPatientDetail from './pages/doctor/DoctorPatientDetail';
import DoctorReports from './pages/doctor/DoctorReports';
import DoctorReportDetail from './pages/doctor/DoctorReportDetail';

// Admin pages
import AdminUsers from './pages/admin/AdminUsers';

// Error pages
import UnauthorizedPage from './pages/UnauthorizedPage';
import NotFoundPage from './pages/NotFoundPage';

import LoadingSpinner from './components/ui/LoadingSpinner';

function HomeRedirect() {
  const { user, loading, isAuthenticated } = useAuth();

  if (loading) return <LoadingSpinner fullPage />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;

  switch (user?.role) {
    case UserRole.Patient:
      return <Navigate to="/patient/dashboard" replace />;
    case UserRole.Doctor:
      return <Navigate to="/doctor/dashboard" replace />;
    case UserRole.Admin:
      return <Navigate to="/admin/users" replace />;
    default:
      return <Navigate to="/login" replace />;
  }
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Home redirect */}
          <Route path="/" element={<HomeRedirect />} />

          {/* Auth routes */}
          <Route element={<AuthLayout />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
          </Route>

          {/* Patient routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<RoleGuard allowedRoles={[UserRole.Patient]} />}>
              <Route element={<AppLayout />}>
                <Route path="/patient/dashboard" element={<PatientDashboard />} />
                <Route path="/patient/profile" element={<PatientProfile />} />
                <Route path="/patient/medical-history" element={<MedicalHistory />} />
                <Route path="/patient/triage" element={<TriageList />} />
                <Route path="/patient/triage/:sessionId" element={<TriageSession />} />
              </Route>
            </Route>
          </Route>

          {/* Doctor routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<RoleGuard allowedRoles={[UserRole.Doctor]} />}>
              <Route element={<AppLayout />}>
                <Route path="/doctor/dashboard" element={<DoctorDashboard />} />
                <Route path="/doctor/patients" element={<DoctorPatients />} />
                <Route path="/doctor/patients/:patientId" element={<DoctorPatientDetail />} />
                <Route path="/doctor/reports" element={<DoctorReports />} />
                <Route path="/doctor/reports/:reportId" element={<DoctorReportDetail />} />
              </Route>
            </Route>
          </Route>

          {/* Admin routes */}
          <Route element={<ProtectedRoute />}>
            <Route element={<RoleGuard allowedRoles={[UserRole.Admin]} />}>
              <Route element={<AppLayout />}>
                <Route path="/admin/users" element={<AdminUsers />} />
              </Route>
            </Route>
          </Route>

          {/* Error pages */}
          <Route path="/unauthorized" element={<UnauthorizedPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
