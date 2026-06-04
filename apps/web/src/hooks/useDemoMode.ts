import { useCallback, useState } from 'react';

const STORAGE_KEY = 'demoMode';

function readStored(): boolean {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === null) return true;
    return s === '1' || s === 'true';
  } catch {
    return true;
  }
}

/** Demo mode hides configuration and runs with DEMO_DEFAULTS on image pick/drop (default ON). */
export function useDemoMode(): [boolean, (next: boolean) => void] {
  const [demo, setDemo] = useState(readStored);

  const setStored = useCallback((next: boolean) => {
    setDemo(next);
    try {
      localStorage.setItem(STORAGE_KEY, next ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, []);

  return [demo, setStored];
}
