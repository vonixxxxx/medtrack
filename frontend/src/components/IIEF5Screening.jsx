import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Heart, AlertCircle, CheckCircle, Info } from 'lucide-react';

const IIEF5Screening = ({ onSubmit, initialData = null }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState(initialData || {
    q1: '', // Confidence in getting erection
    q2: '', // Erection firmness during penetration
    q3: '', // Maintaining erection during intercourse
    q4: '', // Maintaining erection to completion
    q5: ''  // Satisfaction with intercourse
  });
  const [isCompleted, setIsCompleted] = useState(false);
  const [score, setScore] = useState(null);

  const questions = [
    {
      id: 'q1',
      text: 'How do you rate your confidence that you could get and keep an erection?',
      options: [
        { value: 1, label: 'Very low' },
        { value: 2, label: 'Low' },
        { value: 3, label: 'Moderate' },
        { value: 4, label: 'High' },
        { value: 5, label: 'Very high' }
      ]
    },
    {
      id: 'q2',
      text: 'When you had erections with sexual stimulation, how often were your erections hard enough for penetration?',
      options: [
        { value: 1, label: 'Almost never/never' },
        { value: 2, label: 'A few times (much less than half the time)' },
        { value: 3, label: 'Sometimes (about half the time)' },
        { value: 4, label: 'Most times (much more than half the time)' },
        { value: 5, label: 'Almost always/always' }
      ]
    },
    {
      id: 'q3',
      text: 'During sexual intercourse, how often were you able to maintain your erection after you had penetrated (entered) your partner?',
      options: [
        { value: 1, label: 'Almost never/never' },
        { value: 2, label: 'A few times (much less than half the time)' },
        { value: 3, label: 'Sometimes (about half the time)' },
        { value: 4, label: 'Most times (much more than half the time)' },
        { value: 5, label: 'Almost always/always' }
      ]
    },
    {
      id: 'q4',
      text: 'During sexual intercourse, how difficult was it to maintain your erection to completion of intercourse?',
      options: [
        { value: 1, label: 'Extremely difficult' },
        { value: 2, label: 'Very difficult' },
        { value: 3, label: 'Difficult' },
        { value: 4, label: 'Slightly difficult' },
        { value: 5, label: 'Not difficult' }
      ]
    },
    {
      id: 'q5',
      text: 'When you attempted sexual intercourse, how often was it satisfactory for you?',
      options: [
        { value: 1, label: 'Almost never/never' },
        { value: 2, label: 'A few times (much less than half the time)' },
        { value: 3, label: 'Sometimes (about half the time)' },
        { value: 4, label: 'Most times (much more than half the time)' },
        { value: 5, label: 'Almost always/always' }
      ]
    }
  ];

  const getSeverity = (totalScore) => {
    if (totalScore >= 22) return { level: 'No ED', color: 'text-green-600', bgColor: 'bg-green-50', icon: CheckCircle };
    if (totalScore >= 17) return { level: 'Mild ED', color: 'text-yellow-600', bgColor: 'bg-yellow-50', icon: AlertCircle };
    if (totalScore >= 12) return { level: 'Mild to Moderate ED', color: 'text-orange-600', bgColor: 'bg-orange-50', icon: AlertCircle };
    if (totalScore >= 8) return { level: 'Moderate ED', color: 'text-red-600', bgColor: 'bg-red-50', icon: AlertCircle };
    return { level: 'Severe ED', color: 'text-red-800', bgColor: 'bg-red-100', icon: AlertCircle };
  };

  const handleAnswer = (questionId, value) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    }
  };

  const handlePrev = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const handleSubmit = () => {
    const total = Object.values(answers).reduce((sum, value) => sum + (parseInt(value) || 0), 0);
    setScore(total);
    setIsCompleted(true);
    
    if (onSubmit) {
      onSubmit({
        answers,
        totalScore: total,
        severity: getSeverity(total),
        completedAt: new Date().toISOString()
      });
    }
  };

  const canProceed = () => {
    return answers[questions[currentQuestion].id] !== '';
  };

  const canCalculate = () => {
    return Object.values(answers).every(value => value !== '');
  };

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  if (isCompleted) {
    const severity = getSeverity(score);
    const IconComponent = severity.icon;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      >
        <div className="text-center mb-6">
          <IconComponent className={`w-16 h-16 mx-auto mb-4 ${severity.color}`} />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">IIEF-5 Assessment Complete</h2>
          <p className="text-gray-600">Your erectile function assessment results</p>
        </div>

        <div className={`p-6 rounded-xl ${severity.bgColor} mb-6`}>
          <div className="text-center">
            <div className="text-4xl font-bold mb-2">{score}/25</div>
            <div className={`text-xl font-semibold ${severity.color}`}>{severity.level}</div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Score Interpretation:</h3>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>22-25:</span>
              <span className="text-green-600 font-medium">No ED</span>
            </div>
            <div className="flex justify-between">
              <span>17-21:</span>
              <span className="text-yellow-600 font-medium">Mild ED</span>
            </div>
            <div className="flex justify-between">
              <span>12-16:</span>
              <span className="text-orange-600 font-medium">Mild to Moderate ED</span>
            </div>
            <div className="flex justify-between">
              <span>8-11:</span>
              <span className="text-red-600 font-medium">Moderate ED</span>
            </div>
            <div className="flex justify-between">
              <span>5-7:</span>
              <span className="text-red-800 font-medium">Severe ED</span>
            </div>
          </div>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => {
              setIsCompleted(false);
              setCurrentQuestion(0);
              setScore(null);
            }}
            className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Retake Assessment
          </button>
          <button
            onClick={() => window.print()}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Print Results
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg"
    >
      <div className="text-center mb-6">
        <Heart className="w-12 h-12 mx-auto mb-4 text-red-500" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">IIEF-5 Assessment</h2>
        <p className="text-gray-600">Erectile Function Screening Tool</p>
        <p className="text-sm text-gray-500 mt-2">
          Question {currentQuestion + 1} of {questions.length}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Current Question */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          {questions[currentQuestion].text}
        </h3>
        
        <div className="space-y-3">
          {questions[currentQuestion].options.map((option) => (
            <label
              key={option.value}
              className={`flex items-center p-3 border rounded-lg cursor-pointer transition-all ${
                answers[questions[currentQuestion].id] === option.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="radio"
                name={questions[currentQuestion].id}
                value={option.value}
                checked={answers[questions[currentQuestion].id] === option.value}
                onChange={() => handleAnswer(questions[currentQuestion].id, option.value)}
                className="mr-3 text-blue-600"
              />
              <span className="text-gray-700">{option.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={handlePrev}
          disabled={currentQuestion === 0}
          className={`px-4 py-2 rounded-lg transition-colors ${
            currentQuestion === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-gray-600 text-white hover:bg-gray-700'
          }`}
        >
          Previous
        </button>

        <button
          onClick={handleNext}
          disabled={currentQuestion === questions.length - 1}
          className={`px-6 py-2 rounded-md ${
            currentQuestion === questions.length - 1
              ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700'
          }`}
        >
          Next
        </button>
        {currentQuestion === questions.length - 1 && (
          <button
            onClick={handleSubmit}
            className="px-6 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700"
          >
            Submit
          </button>
        )}
      </div>

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start">
          <Info className="w-5 h-5 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-medium mb-1">About IIEF-5</p>
            <p>
              The International Index of Erectile Function (IIEF-5) is a validated 5-item questionnaire 
              used to assess erectile dysfunction severity. This tool helps healthcare providers evaluate 
              and monitor treatment effectiveness.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default IIEF5Screening;
