import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import { MainLayout } from './MainLayout';
import { Sidebar } from './components/Sidebar';
import { useDemoMode } from './hooks/useDemoMode';
import { fetchConfig } from './lib/api';
import { AvailabilityPage } from './pages/AvailabilityPage';

export default function App() {
  const [demoMode, setDemoMode] = useDemoMode();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const configQuery = useQuery({
    queryKey: ['config'],
    queryFn: fetchConfig,
    staleTime: 60_000,
  });

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        setSidebarOpen((o) => !o);
      }
      if (e.key === 'Escape') setSidebarOpen(false);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="relative">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        demoMode={demoMode}
        onDemoChange={setDemoMode}
        config={configQuery.data}
        configLoading={configQuery.isLoading}
        configError={
          configQuery.error instanceof Error
            ? configQuery.error.message
            : undefined
        }
      />
      <Routes>
        <Route
          path="/"
          element={
            <MainLayout
              demoMode={demoMode}
              onOpenSettings={() => setSidebarOpen(true)}
              config={configQuery.data}
            />
          }
        />
        <Route
          path="/availability"
          element={<AvailabilityPage onOpenSettings={() => setSidebarOpen(true)} />}
        />
      </Routes>
    </div>
  );
}
