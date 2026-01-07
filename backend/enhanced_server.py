#!/usr/bin/env python3
"""
Enhanced DevOps AI Assistant Backend with RAG Model
Advanced backend with ChromaDB RAG implementation and training progress tracking
"""

import os
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import anthropic
import re
from collections import Counter
import math
import time
from github import Github, GithubException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DevOps AI Assistant", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LogSearchRequest(BaseModel):
    log_content: str

class GitHubAnalysisRequest(BaseModel):
    github_url: str

class MarkSolutionRequest(BaseModel):
    error_text: str
    solution_text: str
    confidence: Optional[float] = 0.95
    source: Optional[str] = "user_verified"

# Global variables for RAG model and training progress
rag_model = None
chroma_client = None
collection = None
training_progress = {
    "total_entries": 0,
    "training_status": "idle",
    "last_training": None,
    "accuracy": 0.0,
    "model_version": "1.0.0"
}

# API Configuration - use environment variables in production
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    logger.warning("CLAUDE_API_KEY not configured; Claude-powered features may be limited.")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Optional, for higher rate limits

class RAGModel:
    def __init__(self):
        self.setup_chroma()
        
    def text_to_vector(self, text):
        """Simple TF-IDF-like vectorization as fallback"""
        # Clean and tokenize text
        words = re.findall(r'\w+', text.lower())
        word_counts = Counter(words)
        
        # Create a simple vector (using hash for consistent dimensionality)
        vector = [0] * 100  # Fixed size vector
        for word, count in word_counts.items():
            hash_val = hash(word) % 100
            vector[hash_val] += count
        
        # Normalize vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return max(0, dot_product)  # Ensure non-negative
        
    def setup_chroma(self):
        """Initialize ChromaDB for vector storage"""
        global chroma_client, collection
        try:
            # Create ChromaDB client
            chroma_client = chromadb.Client(Settings(
                is_persistent=True,
                persist_directory="./chroma_db"
            ))
            
            # Get or create collection
            collection = chroma_client.get_or_create_collection(
                name="devops_solutions",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_solution(self, error_text: str, solution_text: str, confidence: float = 0.95, source: str = "user_verified"):
        """Add a new solution to the RAG database"""
        try:
            # Generate embedding for error text using simple vectorization
            embedding = self.text_to_vector(error_text)
            
            # Create unique ID
            doc_id = str(uuid.uuid4())
            
            # Add to ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[error_text],
                metadatas=[{
                    "solution": solution_text,
                    "confidence": confidence,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                    "id": doc_id
                }],
                ids=[doc_id]
            )
            
            # Update training progress
            training_progress["total_entries"] = collection.count()
            training_progress["last_training"] = datetime.now().isoformat()
            
            logger.info(f"Added solution to RAG database: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add solution to RAG: {e}")
            raise
    
    def search_solutions(self, query: str, n_results: int = 3, min_similarity: float = 0.7) -> List[Dict]:
        """Search for similar solutions in the RAG database with improved similarity matching"""
        try:
            if collection.count() == 0:
                return []
            
            # Generate embedding for query using simple vectorization
            query_embedding = self.text_to_vector(query)
            
            # Search in ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count())
            )
            
            solutions = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Calculate similarity score (1 - cosine distance)
                    similarity = max(0, 1 - distance)
                    
                    # Use higher threshold for better quality matches
                    # Also check for key terms overlap for better relevance
                    if similarity > min_similarity:
                        # Additional relevance check - compare key terms
                        query_terms = set(re.findall(r'\w+', query.lower()))
                        doc_terms = set(re.findall(r'\w+', doc.lower()))
                        
                        # Calculate term overlap ratio
                        if query_terms:
                            term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms)
                        else:
                            term_overlap = 0
                        
                        # Combine similarity and term overlap for final confidence
                        combined_confidence = (similarity * 0.7) + (term_overlap * 0.3)
                        
                        # Only include if the combined confidence meets our threshold
                        if combined_confidence > 0.5:
                            solutions.append({
                                "id": metadata["id"],
                                "description": f"RAG Match: {doc[:100]}...",
                                "confidence": min(0.98, combined_confidence * metadata.get("confidence", 0.95)),
                                "solution_text": metadata["solution"],
                                "source": "rag",
                                "similarity": similarity,
                                "term_overlap": term_overlap,
                                "combined_confidence": combined_confidence
                            })
            
            # Sort by combined confidence
            solutions.sort(key=lambda x: x.get("combined_confidence", 0), reverse=True)
            return solutions
        except Exception as e:
            logger.error(f"Failed to search RAG database: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get RAG model statistics"""
        try:
            total_entries = collection.count() if collection else 0
            return {
                "total_entries": total_entries,
                "model_version": training_progress["model_version"],
                "last_training": training_progress["last_training"],
                "training_status": training_progress["training_status"],
                "accuracy": training_progress["accuracy"],
                "embedding_model": "simple_tfidf_vectorizer",
                "vector_dimensions": 100
            }
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}

async def get_claude_recommendations(error_text: str) -> List[Dict]:
    """Get recommendations from Claude AI using real Anthropic API"""
    try:
        if not CLAUDE_API_KEY:
            logger.info("CLAUDE_API_KEY not set; returning fallback solutions.")
            return get_fallback_solutions(error_text)
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        
        # Craft a detailed prompt for DevOps troubleshooting
        prompt = f"""You are an expert DevOps engineer. Analyze this error/log and provide specific troubleshooting solutions.

Error/Log content:
{error_text}

Please provide 3-5 practical solutions in JSON format with the following structure:
[
  {{
    "description": "Brief description of the issue",
    "confidence": "confidence score between 0.1-1.0",
    "solution_text": "Detailed step-by-step solution"
  }}
]

Focus on:
- Common DevOps issues (CI/CD, containers, deployment, configuration)
- Actionable steps the user can take
- Real-world troubleshooting approaches
- Consider different scenarios that might cause this error

Provide only the JSON array, no other text."""

        # Make API call to Claude
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Using Haiku for faster responses
            max_tokens=2000,
            temperature=0.3,  # Lower temperature for more consistent technical responses
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse Claude's response
        response_text = message.content[0].text.strip()
        
        # Try to extract JSON from the response
        import json
        import re
        
        # Look for JSON array in the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                solutions_data = json.loads(json_match.group())
                solutions = []
                
                for i, sol in enumerate(solutions_data[:5]):  # Limit to 5 solutions
                    solutions.append({
                        "id": f"claude_{uuid.uuid4()}",
                        "description": sol.get("description", f"Claude AI Solution #{i+1}"),
                        "confidence": float(sol.get("confidence", 0.8)),
                        "solution_text": sol.get("solution_text", "No solution details provided"),
                        "source": "claude"
                    })
                
                if solutions:
                    logger.info(f"Retrieved {len(solutions)} solutions from Claude API")
                    return solutions
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Claude JSON response: {e}")
        
        # If JSON parsing fails, try to extract content anyway
        logger.warning("Claude API returned non-JSON response, parsing as text")
        return [{
            "id": f"claude_{uuid.uuid4()}",
            "description": "Claude AI Analysis",
            "confidence": 0.7,
            "solution_text": response_text,
            "source": "claude"
        }]
        
    except anthropic.AuthenticationError:
        logger.error("Claude API authentication failed - invalid API key")
        return get_fallback_solutions(error_text)
    except anthropic.RateLimitError:
        logger.error("Claude API rate limit exceeded")
        return get_fallback_solutions(error_text)
    except Exception as e:
        logger.error(f"Failed to get Claude recommendations: {e}")
        return get_fallback_solutions(error_text)

def get_fallback_solutions(error_text: str) -> List[Dict]:
    """Provide fallback solutions when Claude API is unavailable"""
    logger.info("Using fallback solutions due to Claude API unavailability")
    
    # Analyze error text for common patterns
    error_lower = error_text.lower()
    solutions = []
    
    if any(keyword in error_lower for keyword in ["npm", "node", "package.json", "module"]):
        solutions.append({
            "id": f"fallback_{uuid.uuid4()}",
            "description": "Node.js/NPM Related Issue",
            "confidence": 0.8,
            "solution_text": "Common Node.js fixes:\n1. Delete node_modules and package-lock.json\n2. Run 'npm install' or 'npm ci'\n3. Check Node.js version compatibility\n4. Verify package.json syntax\n5. Clear npm cache: 'npm cache clean --force'",
            "source": "claude"
        })
    
    if any(keyword in error_lower for keyword in ["docker", "container", "image"]):
        solutions.append({
            "id": f"fallback_{uuid.uuid4()}",
            "description": "Docker/Container Issue",
            "confidence": 0.8,
            "solution_text": "Docker troubleshooting steps:\n1. Check Docker daemon is running\n2. Verify Dockerfile syntax\n3. Check image build context\n4. Ensure sufficient disk space\n5. Try rebuilding without cache: 'docker build --no-cache'",
            "source": "claude"
        })
    
    if any(keyword in error_lower for keyword in ["permission", "denied", "access"]):
        solutions.append({
            "id": f"fallback_{uuid.uuid4()}",
            "description": "Permission/Access Issue",
            "confidence": 0.8,
            "solution_text": "Permission fixes:\n1. Check file/directory permissions\n2. Verify user has necessary access rights\n3. For Docker: check user namespace mapping\n4. For Git: verify SSH keys or tokens\n5. Run with appropriate sudo if needed",
            "source": "claude"
        })
    
    if any(keyword in error_lower for keyword in ["python", "pip", "requirements.txt"]):
        solutions.append({
            "id": f"fallback_{uuid.uuid4()}",
            "description": "Python/Pip Related Issue",
            "confidence": 0.8,
            "solution_text": "Python troubleshooting:\n1. Check Python version compatibility\n2. Update pip: 'pip install --upgrade pip'\n3. Use virtual environment\n4. Verify requirements.txt syntax\n5. Try installing dependencies one by one",
            "source": "claude"
        })
    
    # Generic fallback if no specific patterns match
    if not solutions:
        solutions.append({
            "id": f"fallback_{uuid.uuid4()}",
            "description": "General DevOps Troubleshooting",
            "confidence": 0.6,
            "solution_text": "General troubleshooting steps:\n1. Check system logs for more details\n2. Verify all dependencies are installed\n3. Check environment variables and configuration\n4. Restart services/applications\n5. Check resource availability (disk, memory, CPU)\n6. Review recent changes that might have caused the issue",
            "source": "claude"
        })
    
    return solutions

def extract_github_repo_info(github_url: str) -> tuple:
    """Extract owner and repo from GitHub URL"""
    pattern = r'github\.com/([^/]+)/([^/]+)'
    match = re.search(pattern, github_url)
    if match:
        return match.group(1), match.group(2).replace('.git', '')
    raise ValueError("Invalid GitHub URL format")

def get_github_failed_workflows(owner: str, repo: str) -> List[dict]:
    """Get actual failed GitHub Actions workflows for a specific repository"""
    try:
        # Initialize GitHub client
        g = Github(GITHUB_TOKEN) if GITHUB_TOKEN else Github()
        
        # Get the repository
        repository = g.get_repo(f"{owner}/{repo}")
        
        # Get workflow runs (limit to recent runs)
        workflow_runs = repository.get_workflow_runs()
        
        failed_pipelines = []
        for run in workflow_runs[:10]:  # Get last 10 runs
            if run.conclusion == "failure":
                # Get workflow details
                workflow = repository.get_workflow(run.workflow_id)
                
                # Get failed jobs for this run
                jobs = run.get_jobs()
                error_logs = []
                
                for job in jobs:
                    if job.conclusion == "failure":
                        # Try to get job logs (note: logs might not be available for all repos)
                        try:
                            # For demo, we'll create a summary based on available job info
                            error_summary = f"Job '{job.name}' failed"
                            if job.steps:
                                failed_steps = [step for step in job.steps if step.conclusion == "failure"]
                                if failed_steps:
                                    error_summary += f" at step: {failed_steps[0].name}"
                            error_logs.append(error_summary)
                        except Exception as e:
                            error_logs.append(f"Job '{job.name}' failed - logs unavailable")
                
                failed_pipelines.append({
                    "workflow_name": workflow.name,
                    "run_id": str(run.id),
                    "failure_reason": f"Workflow failed: {run.conclusion}",
                    "logs": "\n".join(error_logs) if error_logs else f"Workflow run #{run.run_number} failed",
                    "run_number": run.run_number,
                    "created_at": run.created_at.isoformat() if run.created_at else datetime.now().isoformat(),
                    "html_url": run.html_url
                })
                
                # Limit to 5 failed workflows to avoid too much data
                if len(failed_pipelines) >= 5:
                    break
        
        # If no failed workflows found, return a message
        if not failed_pipelines:
            failed_pipelines.append({
                "workflow_name": "No Failed Workflows",
                "run_id": "N/A",
                "failure_reason": f"No recent failed workflows found for {owner}/{repo}",
                "logs": "This repository appears to have no recent workflow failures. Great job! 🎉",
                "run_number": 0,
                "created_at": datetime.now().isoformat(),
                "html_url": f"https://github.com/{owner}/{repo}/actions"
            })
        
        return failed_pipelines
        
    except GithubException as e:
        # Be careful with GithubException logging - it can contain bytes
        logger.error(f"GitHub API error status: {e.status}")
        if e.status == 404:
            raise ValueError(f"Repository {owner}/{repo} not found or not accessible")
        elif e.status == 403:
            # Rate limit or permission issue - return a helpful message instead of crashing
            return [{
                "workflow_name": "GitHub API Rate Limited",
                "run_id": "N/A",
                "failure_reason": "GitHub API rate limit exceeded or repository access denied",
                "logs": f"Repository: {owner}/{repo}\nPlease provide a GitHub token for higher API limits or check repository permissions.\nYou can still test with public repositories that have recent workflow failures.",
                "run_number": 0,
                "created_at": datetime.now().isoformat(),
                "html_url": f"https://github.com/{owner}/{repo}/actions"
            }]
        else:
            # Return a user-friendly error instead of crashing
            error_message = f"GitHub API error (status: {e.status})"
            
            return [{
                "workflow_name": "GitHub API Error",
                "run_id": "N/A",
                "failure_reason": error_message,
                "logs": f"Repository: {owner}/{repo}\nPlease check the repository URL and try again.",
                "run_number": 0,
                "created_at": datetime.now().isoformat(),
                "html_url": f"https://github.com/{owner}/{repo}/actions"
            }]
    except Exception as e:
        logger.error(f"Error fetching GitHub workflows: {type(e).__name__}")
        return [{
            "workflow_name": "Error Fetching Workflows",
            "run_id": "N/A",
            "failure_reason": f"Failed to fetch workflows: {type(e).__name__}",
            "logs": "Please check the repository URL and try again",
            "run_number": 0,
            "created_at": datetime.now().isoformat(),
            "html_url": f"https://github.com/{owner}/{repo}/actions"
        }]

async def simulate_training_progress():
    """Simulate training progress for UI feedback"""
    training_progress["training_status"] = "training"
    
    # Simulate training steps
    steps = ["Loading data", "Computing embeddings", "Updating model", "Validating accuracy", "Saving model"]
    for i, step in enumerate(steps):
        training_progress["current_step"] = step
        training_progress["progress_percent"] = (i + 1) * 20
        await asyncio.sleep(0.5)  # Simulate processing time
    
    # Update final status
    training_progress["training_status"] = "idle"
    training_progress["accuracy"] = min(0.95, 0.7 + (training_progress["total_entries"] * 0.01))
    training_progress["last_training"] = datetime.now().isoformat()
    training_progress["current_step"] = "Complete"
    training_progress["progress_percent"] = 100

# Initialize RAG model
try:
    rag_model = RAGModel()
    logger.info("RAG Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG model: {e}")
    rag_model = None

# API Endpoints
@app.get("/")
async def root():
    return {"message": "DevOps AI Assistant Enhanced API v2.0", "rag_enabled": rag_model is not None}

@app.get("/health")
async def health_check():
    """Health check with real API validation"""
    claude_api_valid = False
    github_api_status = "not_configured"
    
    # Test Claude API
    try:
        if CLAUDE_API_KEY and CLAUDE_API_KEY != "":
            client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
            # Make a minimal test call
            test_message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            claude_api_valid = True
    except anthropic.AuthenticationError:
        logger.warning("Claude API authentication failed")
        claude_api_valid = False
    except Exception as e:
        logger.warning(f"Claude API test failed: {e}")
        claude_api_valid = False
    
    # Check GitHub API status
    if GITHUB_TOKEN:
        github_api_status = "configured_with_token"
    else:
        github_api_status = "configured_without_token"
    
    return {
        "status": "healthy",
        "claude_api_valid": claude_api_valid,
        "github_api_status": github_api_status,
        "rag_model_loaded": rag_model is not None,
        "database_status": "connected" if collection else "disconnected"
    }

@app.post("/search-logs")
async def search_logs(request: LogSearchRequest):
    """Search for solutions based on log content with improved RAG and Claude integration"""
    try:
        log_content = request.log_content.strip()
        if not log_content:
            raise HTTPException(status_code=400, detail="Log content cannot be empty")
        
        solutions = []
        
        # First, search RAG database with higher quality threshold
        if rag_model:
            rag_solutions = rag_model.search_solutions(log_content, n_results=5, min_similarity=0.7)
            solutions.extend(rag_solutions)
        
        # If no high-quality RAG solutions found, or only low-confidence ones, get Claude recommendations
        high_confidence_rag = [s for s in solutions if s.get("combined_confidence", 0) > 0.8]
        
        if not high_confidence_rag:
            logger.info("No high-confidence RAG matches found, querying Claude AI")
            claude_solutions = await get_claude_recommendations(log_content)
            
            # If we have some RAG solutions but they're low confidence, add Claude solutions as well
            if solutions:
                # Add Claude solutions but mark them clearly
                for claude_sol in claude_solutions:
                    claude_sol["description"] = f"Claude AI: {claude_sol['description']}"
                solutions.extend(claude_solutions)
            else:
                # No RAG solutions at all, use Claude
                solutions = claude_solutions
        
        # Mark source for each solution and sort by confidence
        for solution in solutions:
            if "source" not in solution:
                solution["source"] = "claude"
        
        # Sort solutions by confidence
        solutions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "solutions": solutions,
            "total_count": len(solutions),
            "sources": {
                "rag": sum(1 for s in solutions if s.get("source") == "rag"),
                "claude": sum(1 for s in solutions if s.get("source") == "claude")
            },
            "search_strategy": "high_confidence_rag" if high_confidence_rag else "claude_fallback"
        }
        
    except Exception as e:
        logger.error(f"Error in search_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-logs")
async def upload_logs(file: UploadFile = File(...)):
    """Upload and analyze log files"""
    try:
        # Validate file type
        if not file.filename.endswith(('.log', '.txt')):
            raise HTTPException(status_code=400, detail="Only .log and .txt files are supported")
        
        # Read file content
        content = await file.read()
        
        # Decode content
        try:
            log_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                log_content = content.decode('latin-1')  # Fallback encoding
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file content")
        
        if not log_content.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        # Parse log content to extract potential errors
        log_content = parse_log_content(log_content)
        
        solutions = []
        
        # Search RAG database with higher quality threshold
        if rag_model:
            rag_solutions = rag_model.search_solutions(log_content, n_results=5, min_similarity=0.7)
            solutions.extend(rag_solutions)
        
        # If no high-quality RAG solutions found, get Claude recommendations
        high_confidence_rag = [s for s in solutions if s.get("combined_confidence", 0) > 0.8]
        
        if not high_confidence_rag:
            logger.info("No high-confidence RAG matches found for uploaded file, querying Claude AI")
            claude_solutions = await get_claude_recommendations(log_content)
            
            if solutions:
                for claude_sol in claude_solutions:
                    claude_sol["description"] = f"Claude AI: {claude_sol['description']}"
                solutions.extend(claude_solutions)
            else:
                solutions = claude_solutions
        
        # Mark source for each solution and sort by confidence
        for solution in solutions:
            if "source" not in solution:
                solution["source"] = "claude"
        
        solutions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "message": f"File '{file.filename}' analyzed successfully",
            "file_size": len(content),
            "parsed_content_preview": log_content[:500] + "..." if len(log_content) > 500 else log_content,
            "solutions": solutions,
            "total_count": len(solutions),
            "sources": {
                "rag": sum(1 for s in solutions if s.get("source") == "rag"),
                "claude": sum(1 for s in solutions if s.get("source") == "claude")
            },
            "search_strategy": "high_confidence_rag" if high_confidence_rag else "claude_fallback"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_log_content(log_content: str) -> str:
    """Parse log content to extract relevant error information"""
    try:
        lines = log_content.split('\n')
        
        # Keywords that typically indicate errors or important log entries
        error_keywords = [
            'error', 'err', 'exception', 'failed', 'failure', 'critical', 'fatal',
            'warn', 'warning', 'denied', 'timeout', 'unable', 'cannot', 'invalid',
            'missing', 'not found', 'permission', 'access', 'refused', 'blocked'
        ]
        
        # Extract lines that contain error keywords or stack traces
        relevant_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for error keywords
            if any(keyword in line_lower for keyword in error_keywords):
                relevant_lines.append(line.strip())
                
                # Include context lines (next 2 lines for stack traces)
                for j in range(1, 3):
                    if i + j < len(lines) and lines[i + j].strip():
                        context_line = lines[i + j].strip()
                        if context_line not in relevant_lines:
                            relevant_lines.append(context_line)
        
        # If no error lines found, return the last 10 lines (often contain the most recent errors)
        if not relevant_lines:
            relevant_lines = [line.strip() for line in lines[-10:] if line.strip()]
        
        # Join the relevant lines
        parsed_content = '\n'.join(relevant_lines)
        
        # If parsed content is too short, include more context
        if len(parsed_content) < 100 and len(log_content) > 100:
            # Take last 1000 characters
            parsed_content = log_content[-1000:]
        
        return parsed_content if parsed_content else log_content
        
    except Exception as e:
        logger.error(f"Error parsing log content: {e}")
        return log_content  # Fallback to original content

@app.post("/analyze-github")
async def analyze_github(request: GitHubAnalysisRequest):
    """Analyze GitHub repository for failed pipelines with improved RAG and Claude integration"""
    try:
        # Extract owner and repo from URL
        owner, repo = extract_github_repo_info(request.github_url)
        
        # Get actual failed workflows from GitHub API
        failed_pipelines = get_github_failed_workflows(owner, repo)
        
        all_results = []
        for pipeline in failed_pipelines:
            error_text = f"{pipeline['failure_reason']}\n{pipeline['logs']}"
            
            solutions = []
            # Search RAG first with higher quality threshold
            if rag_model:
                rag_solutions = rag_model.search_solutions(error_text, n_results=3, min_similarity=0.7)
                solutions.extend(rag_solutions)
            
            # If no high-quality RAG solutions, get Claude recommendations
            high_confidence_rag = [s for s in solutions if s.get("combined_confidence", 0) > 0.8]
            
            if not high_confidence_rag:
                logger.info(f"No high-confidence RAG matches for pipeline {pipeline['workflow_name']}, querying Claude AI")
                claude_solutions = await get_claude_recommendations(error_text)
                
                # Enhance Claude solutions with GitHub Actions context
                for claude_sol in claude_solutions:
                    claude_sol["description"] = f"GitHub Actions: {claude_sol['description']}"
                    claude_sol["solution_text"] = enhance_github_solution(claude_sol["solution_text"], pipeline)
                
                if solutions:
                    solutions.extend(claude_solutions)
                else:
                    solutions = claude_solutions
            
            # Sort by confidence
            solutions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            all_results.append({
                "pipeline": pipeline,
                "solutions": solutions,
                "solution_count": len(solutions),
                "search_strategy": "high_confidence_rag" if high_confidence_rag else "claude_fallback"
            })
        
        return {"results": all_results}
        
    except ValueError as e:
        logger.error(f"GitHub URL parsing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_github: {e}")
        # Return a safe fallback response instead of crashing
        fallback_result = [{
            "workflow_name": "Error Occurred",
            "run_id": "N/A", 
            "failure_reason": f"Failed to analyze repository: {str(e)}",
            "logs": f"Repository: {request.github_url}\nError: {str(e)}",
            "run_number": 0,
            "created_at": datetime.now().isoformat(),
            "html_url": request.github_url
        }]
        
        solutions = await get_claude_recommendations("GitHub analysis error")
        
        return {
            "results": [{
                "pipeline": fallback_result[0],
                "solutions": solutions,
                "solution_count": len(solutions),
                "search_strategy": "error_fallback"
            }]
        }

def enhance_github_solution(solution_text: str, pipeline: dict) -> str:
    """Enhance solution with GitHub Actions specific context"""
    try:
        workflow_name = pipeline.get('workflow_name', 'Unknown Workflow')
        run_number = pipeline.get('run_number', 'N/A')
        
        enhanced_solution = f"""GitHub Actions Workflow: {workflow_name} (Run #{run_number})

{solution_text}

GitHub Actions Specific Steps:
1. Check the workflow file (.github/workflows/*.yml) for syntax errors
2. Verify secrets and environment variables are properly configured
3. Check runner environments and dependencies
4. Review the specific job that failed: {pipeline.get('logs', 'Check logs for details')}
5. Consider re-running the workflow if it's a transient issue

Workflow URL: {pipeline.get('html_url', 'N/A')}
"""
        return enhanced_solution
    except Exception as e:
        logger.error(f"Error enhancing GitHub solution: {e}")
        return solution_text

@app.post("/mark-solution")
async def mark_solution(request: MarkSolutionRequest, background_tasks: BackgroundTasks):
    """Mark a solution as correct and add to RAG database"""
    try:
        if not rag_model:
            raise HTTPException(status_code=500, detail="RAG model not available")
        
        # Add solution to RAG database
        doc_id = rag_model.add_solution(
            error_text=request.error_text,
            solution_text=request.solution_text,
            confidence=request.confidence or 0.95,
            source=request.source or "user_verified"
        )
        
        # Start training simulation in background
        background_tasks.add_task(simulate_training_progress)
        
        return {
            "message": "Solution marked as correct and added to RAG database",
            "doc_id": doc_id,
            "training_started": True
        }
        
    except Exception as e:
        logger.error(f"Error in mark_solution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag-stats")
async def get_rag_stats():
    """Get RAG model statistics and training progress"""
    try:
        if not rag_model:
            return {"error": "RAG model not available"}
        
        stats = rag_model.get_stats()
        stats.update(training_progress)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_rag_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-progress")
async def get_training_progress():
    """Get current training progress"""
    return training_progress

@app.get("/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        rag_stats = rag_model.get_stats() if rag_model else {}
        
        dashboard_data = {
            "overview": {
                "total_solutions": rag_stats.get("total_entries", 0),
                "model_accuracy": training_progress.get("accuracy", 0.0),
                "last_training": training_progress.get("last_training"),
                "model_version": training_progress.get("model_version", "1.0.0")
            },
            "training_status": {
                "status": training_progress.get("training_status", "idle"),
                "current_step": training_progress.get("current_step", "Ready"),
                "progress_percent": training_progress.get("progress_percent", 0)
            },
            "performance_metrics": {
                "rag_solutions_served": rag_stats.get("total_entries", 0) * 2,  # Mock data
                "claude_solutions_served": 150,  # Mock data
                "accuracy_trend": [0.7, 0.75, 0.8, 0.85, training_progress.get("accuracy", 0.9)],
                "solution_sources": {
                    "rag": 60,
                    "claude": 40
                }
            },
            "recent_activity": [
                {"timestamp": datetime.now().isoformat(), "action": "Solution added", "type": "rag"},
                {"timestamp": (datetime.now()).isoformat(), "action": "Training completed", "type": "training"}
            ]
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error in get_dashboard_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)