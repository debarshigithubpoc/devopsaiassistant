import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = '';

function App() {
  const [activeTab, setActiveTab] = useState('logs');
  const [logContent, setLogContent] = useState('');
  const [githubUrl, setGithubUrl] = useState('');
  const [solutions, setSolutions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [ragStats, setRagStats] = useState(null);

  const handleLogSearch = async () => {
    if (!logContent.trim()) {
      setError('Please enter log content');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('/search-logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ log_content: logContent })
      });
      
      const data = await response.json();
      setSolutions(data.solutions);
    } catch (err) {
      setError('Failed to search for solutions');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/upload-logs', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setSolutions(data.solutions);
    } catch (err) {
      setError('Failed to upload and analyze file');
    } finally {
      setLoading(false);
    }
  };

  const handleGithubAnalysis = async () => {
    if (!githubUrl.trim()) {
      setError('Please enter a GitHub URL');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('/analyze-github', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ github_url: githubUrl })
      });
      
      const data = await response.json();
      // Flatten solutions from all pipelines
      const allSolutions = data.results.flatMap(result => 
        result.solutions.map(sol => ({
          ...sol,
          pipeline: result.pipeline.workflow_name
        }))
      );
      setSolutions(allSolutions);
    } catch (err) {
      setError('Failed to analyze GitHub repository');
    } finally {
      setLoading(false);
    }
  };

  const markSolutionCorrect = async (solution) => {
    try {
      await fetch('/mark-solution', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          solution_id: solution.id,
          error_text: activeTab === 'logs' ? logContent : githubUrl,
          solution_text: solution.solution_text
        })
      });
      
      alert('Solution marked as correct and added to the knowledge base!');
    } catch (err) {
      setError('Failed to mark solution as correct');
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 DevOps AI Assistant</h1>
        <p>Intelligent solution recommendations for your DevOps challenges</p>
      </header>

      <div className="main-container">
        <div className="tab-container">
          <button 
            className={`tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            📝 Log Analysis
          </button>
          <button 
            className={`tab ${activeTab === 'github' ? 'active' : ''}`}
            onClick={() => setActiveTab('github')}
          >
            🔧 GitHub Actions
          </button>
        </div>

        <div className="content">
          {activeTab === 'logs' && (
            <div className="tab-content">
              <h2>Developer Log Analysis</h2>
              <p>Search logs or upload log files to get AI-powered solution recommendations</p>
              
              <div className="input-section">
                <div className="search-section">
                  <h3>Search Logs</h3>
                  <textarea
                    value={logContent}
                    onChange={(e) => setLogContent(e.target.value)}
                    placeholder="Paste your error logs here..."
                    rows="6"
                    className="log-textarea"
                  />
                  <button 
                    onClick={handleLogSearch}
                    disabled={loading}
                    className="search-button"
                  >
                    {loading ? '🔍 Analyzing...' : '🔍 Search Solutions'}
                  </button>
                </div>

                <div className="upload-section">
                  <h3>Upload Log File</h3>
                  <input
                    type="file"
                    accept=".log,.txt"
                    onChange={handleFileUpload}
                    className="file-input"
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'github' && (
            <div className="tab-content">
              <h2>GitHub Actions Analysis</h2>
              <p>Analyze failed GitHub Actions pipelines and get solution recommendations</p>
              
              <div className="input-section">
                <h3>GitHub Repository URL</h3>
                <input
                  type="url"
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  placeholder="https://github.com/owner/repository"
                  className="github-input"
                />
                <button 
                  onClick={handleGithubAnalysis}
                  disabled={loading}
                  className="search-button"
                >
                  {loading ? '🔍 Analyzing...' : '🔍 Analyze Pipelines'}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {solutions.length > 0 && (
            <div className="solutions-section">
              <h3>💡 Recommended Solutions</h3>
              {solutions.map((solution, index) => (
                <div key={index} className="solution-card">
                  <div className="solution-header">
                    <h4>{solution.description}</h4>
                    <div className="confidence-badge">
                      Confidence: {Math.round(solution.confidence * 100)}%
                    </div>
                    {solution.pipeline && (
                      <div className="pipeline-badge">
                        Pipeline: {solution.pipeline}
                      </div>
                    )}
                  </div>
                  <div className="solution-content">
                    <pre>{solution.solution_text}</pre>
                  </div>
                  <button 
                    onClick={() => markSolutionCorrect(solution)}
                    className="mark-correct-button"
                  >
                    ✅ Mark as Correct Solution
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;