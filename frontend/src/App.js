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

  // Load dashboard data and RAG stats on component mount
  useEffect(() => {
    loadDashboardData();
    loadRagStats();
    
    // Set up intervals for real-time updates
    const dashboardInterval = setInterval(loadDashboardData, 5000);
    const progressInterval = setInterval(loadTrainingProgress, 1000);
    
    return () => {
      clearInterval(dashboardInterval);
      clearInterval(progressInterval);
    };
  }, []);

  const loadDashboardData = async () => {
    try {
      const response = await fetch('/dashboard-data');
      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    }
  };

  const loadRagStats = async () => {
    try {
      const response = await fetch('/rag-stats');
      if (response.ok) {
        const data = await response.json();
        setRagStats(data);
      }
    } catch (err) {
      console.error('Failed to load RAG stats:', err);
    }
  };

  const loadTrainingProgress = async () => {
    try {
      const response = await fetch('/training-progress');
      if (response.ok) {
        const data = await response.json();
        setTrainingProgress(data);
      }
    } catch (err) {
      console.error('Failed to load training progress:', err);
    }
  };

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
      const response = await fetch('/mark-solution', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          error_text: activeTab === 'logs' ? logContent : githubUrl,
          solution_text: solution.solution_text,
          confidence: solution.confidence,
          source: 'user_verified'
        })
      });
      
      if (response.ok) {
        alert('✅ Solution marked as correct and added to RAG database! Training started...');
        // Refresh stats after marking solution
        setTimeout(() => {
          loadRagStats();
          loadDashboardData();
        }, 1000);
      }
    } catch (err) {
      setError('Failed to mark solution as correct');
    }
  };

  const DashboardTab = () => (
    <div className="dashboard-container">
      <h2>🎯 RAG AI Model Dashboard</h2>
      
      {/* Overview Cards */}
      <div className="stats-grid">
        <div className="stat-card overview">
          <h3>📊 Model Overview</h3>
          <div className="stat-value">{dashboardData?.overview?.total_solutions || 0}</div>
          <div className="stat-label">Total Solutions in RAG Database</div>
          <div className="stat-accuracy">
            Accuracy: {Math.round((dashboardData?.overview?.model_accuracy || 0) * 100)}%
          </div>
        </div>
        
        <div className="stat-card performance">
          <h3>⚡ Performance</h3>
          <div className="performance-metric">
            <span className="metric-label">RAG Solutions Served:</span>
            <span className="metric-value">{dashboardData?.performance_metrics?.rag_solutions_served || 0}</span>
          </div>
          <div className="performance-metric">
            <span className="metric-label">Claude Solutions Served:</span>
            <span className="metric-value">{dashboardData?.performance_metrics?.claude_solutions_served || 0}</span>
          </div>
        </div>
        
        <div className="stat-card training">
          <h3>🤖 Training Status</h3>
          <div className="training-status">
            <div className={`status-indicator ${trainingProgress?.training_status || 'idle'}`}>
              {trainingProgress?.training_status === 'training' ? '🔄 Training' : '✅ Ready'}
            </div>
            {trainingProgress?.training_status === 'training' && (
              <div className="training-details">
                <div className="current-step">{trainingProgress?.current_step}</div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${trainingProgress?.progress_percent || 0}%` }}
                  ></div>
                </div>
                <div className="progress-text">{trainingProgress?.progress_percent || 0}%</div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Solution Sources Chart */}
      <div className="chart-section">
        <h3>🎯 Solution Sources Distribution</h3>
        <div className="source-chart">
          <div className="source-item rag">
            <div className="source-bar">
              <div 
                className="source-fill rag-fill" 
                style={{ 
                  width: `${dashboardData?.performance_metrics?.solution_sources?.rag || 60}%` 
                }}
              ></div>
            </div>
            <span className="source-label">
              RAG Database ({dashboardData?.performance_metrics?.solution_sources?.rag || 60}%)
            </span>
          </div>
          <div className="source-item claude">
            <div className="source-bar">
              <div 
                className="source-fill claude-fill" 
                style={{ 
                  width: `${dashboardData?.performance_metrics?.solution_sources?.claude || 40}%` 
                }}
              ></div>
            </div>
            <span className="source-label">
              Claude AI ({dashboardData?.performance_metrics?.solution_sources?.claude || 40}%)
            </span>
          </div>
        </div>
      </div>
      
      {/* Accuracy Trend */}
      <div className="trend-section">
        <h3>📈 Model Accuracy Trend</h3>
        <div className="trend-chart">
          {(dashboardData?.performance_metrics?.accuracy_trend || []).map((accuracy, index) => (
            <div key={index} className="trend-bar">
              <div 
                className="trend-fill" 
                style={{ height: `${accuracy * 100}%` }}
              ></div>
              <span className="trend-label">{Math.round(accuracy * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Recent Activity */}
      <div className="activity-section">
        <h3>📋 Recent Activity</h3>
        <div className="activity-list">
          {(dashboardData?.recent_activity || []).map((activity, index) => (
            <div key={index} className={`activity-item ${activity.type}`}>
              <span className="activity-time">
                {new Date(activity.timestamp).toLocaleTimeString()}
              </span>
              <span className="activity-action">{activity.action}</span>
              <span className={`activity-type ${activity.type}`}>
                {activity.type === 'rag' ? '🧠 RAG' : '🤖 Training'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 DevOps AI Assistant</h1>
        <p>Intelligent solution recommendations powered by RAG AI & Claude</p>
        {ragStats && (
          <div className="header-stats">
            <span className="stat-item">
              📚 {ragStats.total_entries} Solutions in RAG DB
            </span>
            <span className="stat-item">
              🎯 {Math.round((ragStats.accuracy || 0) * 100)}% Accuracy
            </span>
            {trainingProgress?.training_status === 'training' && (
              <span className="stat-item training-indicator">
                🔄 Training in Progress...
              </span>
            )}
          </div>
        )}
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
          <button 
            className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 RAG Dashboard
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

          {activeTab === 'dashboard' && <DashboardTab />}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {solutions.length > 0 && activeTab !== 'dashboard' && (
            <div className="solutions-section">
              <h3>💡 Recommended Solutions</h3>
              {solutions.map((solution, index) => (
                <div key={index} className={`solution-card ${solution.source}`}>
                  <div className="solution-header">
                    <h4>{solution.description}</h4>
                    <div className="solution-badges">
                      <div className="confidence-badge">
                        Confidence: {Math.round(solution.confidence * 100)}%
                      </div>
                      <div className={`source-badge ${solution.source}`}>
                        {solution.source === 'rag' ? '🧠 RAG Database' : '🤖 Claude AI'}
                      </div>
                      {solution.pipeline && (
                        <div className="pipeline-badge">
                          Pipeline: {solution.pipeline}
                        </div>
                      )}
                    </div>
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