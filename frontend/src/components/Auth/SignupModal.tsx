import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import styles from './Auth.module.css';

interface SignupModalProps {
  onClose: () => void;
  onSwitchToSignin: () => void;
}

const SignupModal: React.FC<SignupModalProps> = ({ onClose, onSwitchToSignin }) => {
  const { signup } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    software_level: 'Intermediate',
    languages_known: [] as string[],
    ai_experience: 'Basic',
    laptop_specs: '',
    gpu_available: 'No GPU',
    robotics_hardware: [] as string[],
    learning_goals: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleMultiSelect = (name: string, value: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      [name]: checked
        ? [...prev[name as keyof typeof formData] as string[], value]
        : (prev[name as keyof typeof formData] as string[]).filter(v => v !== value)
    }));
  };

  const handleNext = () => {
    if (step === 1) {
      if (!formData.email || !formData.password) {
        setError('Email and password are required');
        return;
      }
      if (formData.password.length < 8) {
        setError('Password must be at least 8 characters');
        return;
      }
      if (formData.password !== formData.confirmPassword) {
        setError('Passwords do not match');
        return;
      }
      setError('');
      setStep(2);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const { confirmPassword, ...signupData } = formData;
      await signup(signupData);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Sign Up - Step {step} of 2</h2>
          <button className={styles.closeButton} onClick={onClose}>
            ✕
          </button>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        {step === 1 && (
          <div className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="email">Email *</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="your@email.com"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="password">Password *</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="••••••••"
                minLength={8}
              />
              <small>Minimum 8 characters</small>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="confirmPassword">Confirm Password *</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                placeholder="••••••••"
              />
            </div>

            <button
              type="button"
              className={styles.submitButton}
              onClick={handleNext}
            >
              Next: Profile Setup
            </button>
          </div>
        )}

        {step === 2 && (
          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="software_level">Software Background *</label>
              <select
                id="software_level"
                name="software_level"
                value={formData.software_level}
                onChange={handleChange}
                required
              >
                <option value="Beginner">Beginner</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Advanced">Advanced</option>
              </select>
            </div>

            <div className={styles.formGroup}>
              <label>Languages Known</label>
              <div className={styles.checkboxGroup}>
                {['Python', 'JavaScript', 'C++', 'ROS', 'CUDA'].map(lang => (
                  <label key={lang} className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={formData.languages_known.includes(lang)}
                      onChange={(e) => handleMultiSelect('languages_known', lang, e.target.checked)}
                    />
                    {lang}
                  </label>
                ))}
              </div>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="ai_experience">AI/ML Experience *</label>
              <select
                id="ai_experience"
                name="ai_experience"
                value={formData.ai_experience}
                onChange={handleChange}
                required
              >
                <option value="None">None</option>
                <option value="Basic">Basic</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Advanced">Advanced</option>
              </select>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="laptop_specs">Laptop/Desktop Specs</label>
              <input
                type="text"
                id="laptop_specs"
                name="laptop_specs"
                value={formData.laptop_specs}
                onChange={handleChange}
                placeholder="e.g., 16GB RAM, Intel i7"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="gpu_available">GPU Available</label>
              <select
                id="gpu_available"
                name="gpu_available"
                value={formData.gpu_available}
                onChange={handleChange}
              >
                <option value="No GPU">No GPU</option>
                <option value="RTX">RTX GPU</option>
              </select>
            </div>

            <div className={styles.formGroup}>
              <label>Robotics Hardware</label>
              <div className={styles.checkboxGroup}>
                {['Jetson Nano', 'Jetson Orin', 'Arduino', 'Raspberry Pi'].map(hw => (
                  <label key={hw} className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={formData.robotics_hardware.includes(hw)}
                      onChange={(e) => handleMultiSelect('robotics_hardware', hw, e.target.checked)}
                    />
                    {hw}
                  </label>
                ))}
              </div>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="learning_goals">Learning Goals (Optional)</label>
              <textarea
                id="learning_goals"
                name="learning_goals"
                value={formData.learning_goals}
                onChange={handleChange}
                placeholder="What do you want to learn from this book?"
                rows={3}
              />
            </div>

            <div className={styles.buttonGroup}>
              <button
                type="button"
                className={styles.backButton}
                onClick={() => setStep(1)}
              >
                Back
              </button>
              <button
                type="submit"
                className={styles.submitButton}
                disabled={loading}
              >
                {loading ? 'Creating Account...' : 'Create Account'}
              </button>
            </div>
          </form>
        )}

        {step === 1 && (
          <div className={styles.switchAuth}>
            Already have an account?{' '}
            <button onClick={onSwitchToSignin} className={styles.linkButton}>
              Sign In
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignupModal;
