import { useState, useEffect } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { BookOpen, Plus, Calendar, Tag, Edit2, Trash2 } from 'lucide-react';
import DashboardCard from '../DashboardCard';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { LoadingSkeleton } from '../dashboard/LoadingSkeleton';
import { EmptyState } from '../dashboard/EmptyState';
import { getDiaryEntries, createDiaryEntry, updateDiaryEntry, deleteDiaryEntry } from '../../api';
import { useAuth } from '../../contexts/AuthContext';

export const DiaryEntry = ({ patientId, selectedDate }) => {
  const prefersReducedMotion = useReducedMotion();
  const { user } = useAuth();
  const [entries, setEntries] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    date: selectedDate || new Date().toISOString().split('T')[0],
    entryType: 'note',
    title: '',
    content: '',
    tags: []
  });
  const [tagInput, setTagInput] = useState('');

  useEffect(() => {
    if (user?.id) {
      loadEntries();
    }
  }, [user, patientId, selectedDate]);

  const loadEntries = async () => {
    try {
      setIsLoading(true);
      const params = {
        userId: user.id,
        patientId: patientId || null
      };
      if (selectedDate) {
        params.startDate = selectedDate;
        params.endDate = selectedDate;
      }
      const data = await getDiaryEntries(params);
      // Ensure data is always an array
      const entriesArray = Array.isArray(data) ? data : [];
      setEntries(entriesArray);
    } catch (error) {
      console.error('Error loading diary entries:', error);
      setEntries([]); // Set empty array on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const entryData = {
        userId: user.id,
        patientId: patientId || null,
        ...formData,
        tags: formData.tags.length > 0 ? JSON.stringify(formData.tags) : null
      };

      if (editingId) {
        await updateDiaryEntry(editingId, entryData);
      } else {
        await createDiaryEntry(entryData);
      }
      setIsFormOpen(false);
      setEditingId(null);
      setFormData({
        date: new Date().toISOString().split('T')[0],
        entryType: 'note',
        title: '',
        content: '',
        tags: []
      });
      loadEntries();
    } catch (error) {
      console.error('Error saving diary entry:', error);
    }
  };

  const handleEdit = (entry) => {
    setEditingId(entry.id);
    setFormData({
      date: new Date(entry.date).toISOString().split('T')[0],
      entryType: entry.entryType,
      title: entry.title || '',
      content: entry.content || '',
      tags: entry.tags ? (typeof entry.tags === 'string' ? JSON.parse(entry.tags) : entry.tags) : []
    });
    setIsFormOpen(true);
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this entry?')) return;
    try {
      await deleteDiaryEntry(id);
      loadEntries();
    } catch (error) {
      console.error('Error deleting diary entry:', error);
    }
  };

  const addTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        tags: [...formData.tags, tagInput.trim()]
      });
      setTagInput('');
    }
  };

  const removeTag = (tag) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter(t => t !== tag)
    });
  };

  const entryTypeColors = {
    mood: 'bg-purple-100 text-purple-700',
    symptom: 'bg-red-100 text-red-700',
    note: 'bg-blue-100 text-blue-700',
    custom: 'bg-neutral-100 text-neutral-700'
  };

  return (
    <DashboardCard
      title="Health Diary"
      icon={<BookOpen size={20} />}
      variant="patient"
      action={
        <Button
          onClick={() => {
            setIsFormOpen(true);
            setEditingId(null);
            setFormData({
              date: selectedDate || new Date().toISOString().split('T')[0],
              entryType: 'note',
              title: '',
              content: '',
              tags: []
            });
          }}
          variant="primary"
          size="sm"
        >
          <Plus size={16} className="mr-1.5" />
          Add Entry
        </Button>
      }
    >
      {isLoading ? (
        <LoadingSkeleton variant="list" count={3} />
      ) : (
        <>
          {entries.length === 0 ? (
            <EmptyState
              icon={BookOpen}
              title="No diary entries yet"
              description="Track your symptoms, mood, and health notes"
              action={{
                label: "Add Entry",
                onClick: () => setIsFormOpen(true)
              }}
            />
          ) : (
            <div className="space-y-3">
              {Array.isArray(entries) && entries.map((entry, index) => (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-4 bg-white rounded-2xl border border-neutral-200 hover:border-primary-300 hover:shadow-medium transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge className={entryTypeColors[entry.entryType] || entryTypeColors.note}>
                          {entry.entryType}
                        </Badge>
                        {entry.title && (
                          <h4 className="font-semibold text-neutral-900">{entry.title}</h4>
                        )}
                        <div className="flex items-center gap-1 text-xs text-neutral-500 ml-auto">
                          <Calendar className="w-3 h-3" />
                          <span>{new Date(entry.date).toLocaleDateString()}</span>
                        </div>
                      </div>
                      {entry.content && (
                        <p className="text-sm text-neutral-700 mb-2">{entry.content}</p>
                      )}
                      {entry.tags && (
                        <div className="flex items-center gap-2 flex-wrap">
                          {(typeof entry.tags === 'string' ? JSON.parse(entry.tags) : entry.tags).map((tag, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              <Tag className="w-3 h-3 mr-1" />
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={() => handleEdit(entry)}
                        variant="ghost"
                        size="sm"
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button
                        onClick={() => handleDelete(entry.id)}
                        variant="ghost"
                        size="sm"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {/* Add/Edit Form */}
          {isFormOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 p-6 bg-neutral-50 rounded-2xl border border-neutral-200"
            >
              <h4 className="font-semibold text-neutral-900 mb-4">
                {editingId ? 'Edit Entry' : 'Add Diary Entry'}
              </h4>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                      Date *
                    </label>
                    <Input
                      type="date"
                      value={formData.date}
                      onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-700 mb-2">
                      Type *
                    </label>
                    <select
                      value={formData.entryType}
                      onChange={(e) => setFormData({ ...formData, entryType: e.target.value })}
                      className="w-full h-11 rounded-md border border-neutral-200 px-4 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                      required
                    >
                      <option value="note">Note</option>
                      <option value="mood">Mood</option>
                      <option value="symptom">Symptom</option>
                      <option value="custom">Custom</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Title
                  </label>
                  <Input
                    value={formData.title}
                    onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                    placeholder="Optional title"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Content *
                  </label>
                  <textarea
                    value={formData.content}
                    onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                    className="w-full h-32 rounded-md border border-neutral-200 px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-primary-500"
                    placeholder="Describe your symptoms, mood, or notes..."
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Tags
                  </label>
                  <div className="flex items-center gap-2 mb-2">
                    <Input
                      value={tagInput}
                      onChange={(e) => setTagInput(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          addTag();
                        }
                      }}
                      placeholder="Add a tag and press Enter"
                    />
                    <Button type="button" onClick={addTag} variant="secondary" size="sm">
                      Add
                    </Button>
                  </div>
                  {formData.tags.length > 0 && (
                    <div className="flex items-center gap-2 flex-wrap">
                      {formData.tags.map((tag, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {tag}
                          <button
                            type="button"
                            onClick={() => removeTag(tag)}
                            className="ml-1 hover:text-error-600"
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <Button type="submit" variant="primary" size="md">
                    {editingId ? 'Update' : 'Add'} Entry
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="md"
                    onClick={() => {
                      setIsFormOpen(false);
                      setEditingId(null);
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </motion.div>
          )}
        </>
      )}
    </DashboardCard>
  );
};

