import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import SigninModal from './SigninModal';
import SignupModal from './SignupModal';
import ProfileModal from './ProfileModal';
import styles from './AuthButton.module.css';

const AuthButton: React.FC = () => {
  const { user, loading } = useAuth();
  const [showSignin, setShowSignin] = useState(false);
  const [showSignup, setShowSignup] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  if (loading) {
    return null;
  }

  return (
    <>
      {user ? (
        <button
          className={styles.authButton}
          onClick={() => setShowProfile(true)}
          title="View Profile"
        >
          <span className={styles.userIcon}>👤</span>
          <span className={styles.userEmail}>{user.email}</span>
        </button>
      ) : (
        <div className={styles.authButtons}>
          <button
            className={styles.signinButton}
            onClick={() => setShowSignin(true)}
          >
            Sign In
          </button>
          <button
            className={styles.signupButton}
            onClick={() => setShowSignup(true)}
          >
            Sign Up
          </button>
        </div>
      )}

      {showSignin && (
        <SigninModal
          onClose={() => setShowSignin(false)}
          onSwitchToSignup={() => {
            setShowSignin(false);
            setShowSignup(true);
          }}
        />
      )}

      {showSignup && (
        <SignupModal
          onClose={() => setShowSignup(false)}
          onSwitchToSignin={() => {
            setShowSignup(false);
            setShowSignin(true);
          }}
        />
      )}

      {showProfile && user && (
        <ProfileModal onClose={() => setShowProfile(false)} />
      )}
    </>
  );
};

export default AuthButton;
