import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import styles from './Auth.module.css';

interface ProfileModalProps {
  onClose: () => void;
}

const ProfileModal: React.FC<ProfileModalProps> = ({ onClose }) => {
  const { user, updateProfile, signout } = useAuth();
  const [formData, setFormData] = useState({
    software_level: '',
    languages_known: [] as string[],
    ai_experience: '',
    laptop_specs: '',
    gpu_available: '',
    robotics_hardware: [] as string[],
    learning_goals: '',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user?.profile) {
      setFormData({
        software_level: user.profile.software_level,
        languages_known: user.profile.languages_known || [],
        ai_experience: user.profile.ai_experience,
        laptop_specs: user.profile.laptop_specs || '',
        gpu_available: user.profile.gpu_available || 'No GPU',
        robotics_hardware: user.profile.robotics_hardware || [],
        learning_goals: user.profile.learning_goals || '',
      });
    }
  }, [user]);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      await updateProfile(formData);
      setSuccess('Profile updated successfully!');
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err: any) {
      setError(err.message || 'Profile update failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSignout = () => {
    signout();
    onClose();
  };

  if (!user) {
    return null;
  }

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Profile Settings</h2>
          <button className={styles.closeButton} onClick={onClose}>
            ✕
          </button>
        </div>

        {error && <div className={styles.error}>{error}</div>}
        {success && <div className={styles.success}>{success}</div>}

        <div className={styles.userInfo}>
          <p><strong>Email:</strong> {user.email}</p>
          <p><strong>Member since:</strong> {new Date(user.created_at).toLocaleDateString()}</p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="software_level">Software Background</label>
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
            <label htmlFor="ai_experience">AI/ML Experience</label>
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
            <label htmlFor="learning_goals">Learning Goals</label>
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
              onClick={handleSignout}
            >
              Sign Out
            </button>
            <button
              type="submit"
              className={styles.submitButton}
              disabled={loading}
            >
              {loading ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ProfileModal;
