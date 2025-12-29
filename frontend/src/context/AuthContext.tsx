import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface UserProfile {
  software_level: string;
  languages_known: string[];
  ai_experience: string;
  laptop_specs?: string;
  gpu_available?: string;
  robotics_hardware: string[];
  learning_goals?: string;
  created_at: string;
  updated_at: string;
}

interface User {
  id: number;
  email: string;
  is_active: boolean;
  created_at: string;
  profile?: UserProfile;
}

interface AuthContextType {
  user: User | null;
  accessToken: string | null;
  loading: boolean;
  signin: (email: string, password: string) => Promise<void>;
  signup: (signupData: SignupData) => Promise<void>;
  signout: () => void;
  updateProfile: (profileData: Partial<UserProfile>) => Promise<void>;
}

interface SignupData {
  email: string;
  password: string;
  software_level: string;
  languages_known: string[];
  ai_experience: string;
  laptop_specs?: string;
  gpu_available?: string;
  robotics_hardware: string[];
  learning_goals?: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load auth state from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('access_token');
    const storedUser = localStorage.getItem('user');

    if (storedToken && storedUser) {
      setAccessToken(storedToken);
      setUser(JSON.parse(storedUser));
    }

    setLoading(false);
  }, []);

  const signin = async (email: string, password: string) => {
    const response = await fetch(`${API_BASE_URL}/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signin failed');
    }

    const data = await response.json();
    setAccessToken(data.access_token);
    setUser(data.user);

    // Store in localStorage
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
  };

  const signup = async (signupData: SignupData) => {
    const response = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(signupData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    const data = await response.json();
    setAccessToken(data.access_token);
    setUser(data.user);

    // Store in localStorage
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
  };

  const signout = () => {
    setAccessToken(null);
    setUser(null);
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  };

  const updateProfile = async (profileData: Partial<UserProfile>) => {
    if (!accessToken) {
      throw new Error('Not authenticated');
    }

    const response = await fetch(`${API_BASE_URL}/auth/profile`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
      },
      body: JSON.stringify(profileData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Profile update failed');
    }

    const updatedProfile = await response.json();

    // Update user with new profile
    if (user) {
      const updatedUser = {
        ...user,
        profile: updatedProfile,
      };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
    }
  };

  const value = {
    user,
    accessToken,
    loading,
    signin,
    signup,
    signout,
    updateProfile,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
