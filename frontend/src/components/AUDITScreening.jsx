import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Droplets, AlertTriangle, CheckCircle, Info, AlertCircle } from 'lucide-react';

const AUDITScreening = ({ onSubmit, initialData = null }) => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState(initialData || {
    q1: '', // Frequency of drinking
    q2: '', // Number of drinks on typical day
    q3: '', // Frequency of 6+ drinks
    q4: '', // Frequency of not being able to stop
    q5: '', // Frequency of failing to do expected
    q6: '', // Frequency of needing drink after heavy drinking
    q7: '', // Frequency of guilt/remorse
    q8: '', // Frequency of being unable to remember
    q9: '', // Injury to self or others
    q10: '' // Concern from others
  });
  const [isCompleted, setIsCompleted] = useState(false);
  const [score, setScore] = useState(null);

  const questions = [
    {
      id: 'q1',
      text: 'How often do you have a drink containing alcohol?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Monthly or less' },
        { value: 2, label: '2-4 times a month' },
        { value: 3, label: '2-3 times a week' },
        { value: 4, label: '4 or more times a week' }
      ]
    },
    {
      id: 'q2',
      text: 'How many drinks containing alcohol do you have on a typical day when you are drinking?',
      options: [
        { value: 0, label: '1 or 2' },
        { value: 1, label: '3 or 4' },
        { value: 2, label: '5 or 6' },
        { value: 3, label: '7, 8, or 9' },
        { value: 4, label: '10 or more' }
      ]
    },
    {
      id: 'q3',
      text: 'How often do you have six or more drinks on one occasion?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q4',
      text: 'How often during the last year have you found that you were not able to stop drinking once you had started?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q5',
      text: 'How often during the last year have you failed to do what was normally expected from you because of drinking?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q6',
      text: 'How often during the last year have you needed a first drink in the morning to get yourself going after a heavy drinking session?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q7',
      text: 'How often during the last year have you had a feeling of guilt or remorse after drinking?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q8',
      text: 'How often during the last year have you been unable to remember what happened the night before because you had been drinking?',
      options: [
        { value: 0, label: 'Never' },
        { value: 1, label: 'Less than monthly' },
        { value: 2, label: 'Monthly' },
        { value: 3, label: 'Weekly' },
        { value: 4, label: 'Daily or almost daily' }
      ]
    },
    {
      id: 'q9',
      text: 'Have you or someone else been injured as a result of your drinking?',
      options: [
        { value: 0, label: 'No' },
        { value: 2, label: 'Yes, but not in the last year' },
        { value: 4, label: 'Yes, during the last year' }
      ]
    },
    {
      id: 'q10',
      text: 'Has a relative or friend, doctor or other health worker been concerned about your drinking or suggested you cut down?',
      options: [
        { value: 0, label: 'No' },
        { value: 2, label: 'Yes, but not in the last year' },
        { value: 4, label: 'Yes, during the last year' }
      ]
    }
  ];

  const getRiskCategory = (totalScore) => {
    if (totalScore <= 7) return { level: 'Low Risk', color: 'text-green-600', bgColor: 'bg-green-50', icon: CheckCircle };
    if (totalScore <= 15) return { level: 'Hazardous Drinking', color: 'text-yellow-600', bgColor: 'bg-yellow-50', icon: AlertTriangle };
    if (totalScore <= 19) return { level: 'Harmful Drinking', color: 'text-orange-600', bgColor: 'bg-orange-50', icon: AlertCircle };
    return { level: 'Possible Dependence', color: 'text-red-800', bgColor: 'bg-red-100', icon: AlertCircle };
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
        riskCategory: getRiskCategory(total),
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
    const riskCategory = getRiskCategory(score);
    const IconComponent = riskCategory.icon;
    
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-lg"
      >
        <div className="text-center mb-6">
          <IconComponent className={`w-16 h-16 mx-auto mb-4 ${riskCategory.color}`} />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">AUDIT Assessment Complete</h2>
          <p className="text-gray-600">Your alcohol use screening results</p>
        </div>

        <div className={`p-6 rounded-xl ${riskCategory.bgColor} mb-6`}>
          <div className="text-center">
            <div className="text-4xl font-bold mb-2">{score}/40</div>
            <div className={`text-xl font-semibold ${riskCategory.color}`}>{riskCategory.level}</div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Score Interpretation:</h3>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>0-7:</span>
              <span className="text-green-600 font-medium">Low Risk</span>
            </div>
            <div className="flex justify-between">
              <span>8-15:</span>
              <span className="text-yellow-600 font-medium">Hazardous Drinking</span>
            </div>
            <div className="flex justify-between">
              <span>16-19:</span>
              <span className="text-orange-600 font-medium">Harmful Drinking</span>
            </div>
            <div className="flex justify-between">
              <span>20+:</span>
              <span className="text-red-800 font-medium">Possible Dependence</span>
            </div>
          </div>
        </div>

        {/* Recommendations based on score */}
        <div className="bg-blue-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-blue-800 mb-2">Recommendations:</h3>
          {score <= 7 && (
            <p className="text-blue-700 text-sm">
              Your alcohol consumption appears to be within low-risk limits. Continue to monitor your drinking patterns.
            </p>
          )}
          {score >= 8 && score <= 15 && (
            <p className="text-yellow-700 text-sm">
              Consider reducing your alcohol intake. This level of consumption may pose health risks over time.
            </p>
          )}
          {score >= 16 && (
            <p className="text-red-700 text-sm">
              Your alcohol consumption may be harmful. Consider speaking with a healthcare provider about your drinking habits.
            </p>
          )}
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
        <Droplets className="w-12 h-12 mx-auto mb-4 text-blue-500" />
        <h2 className="text-2xl font-bold text-gray-800 mb-2">AUDIT Assessment</h2>
        <p className="text-gray-600">Alcohol Use Disorders Identification Test</p>
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
            <p className="font-medium mb-1">About AUDIT</p>
            <p>
              The Alcohol Use Disorders Identification Test (AUDIT) is a 10-question screening tool 
              developed by the World Health Organization to detect risky alcohol consumption patterns. 
              This assessment helps identify potential alcohol-related health issues.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default AUDITScreening;
