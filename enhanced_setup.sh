#!/bin/bash

# DevOps AI Assistant - Enhanced Setup Script
# Sets up the complete RAG AI enhanced DevOps AI Assistant

echo "🚀 Setting up DevOps AI Assistant with RAG AI Enhancement..."

# Create necessary directories
mkdir -p backend/chroma_db
mkdir -p tmp

# Setup Backend
echo "📦 Setting up backend..."
cd backend

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Backend setup complete!"

# Setup Frontend
echo "📦 Setting up frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"

cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating environment configuration file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before running the application"
fi

echo "🎯 Setup completed successfully!"
echo ""
echo "🔧 IMPORTANT: Configure API Keys"
echo "   Edit the .env file and add your API keys:"
echo "   - CLAUDE_API_KEY: Get from https://console.anthropic.com/"
echo "   - GITHUB_TOKEN: Get from https://github.com/settings/tokens (optional)"
echo ""
echo "🚀 To start the application:"
echo ""
echo "1. Start the backend (in one terminal):"
echo "   cd backend && python enhanced_server.py"
echo ""
echo "2. Start the frontend (in another terminal):"
echo "   cd frontend && npm start"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "🧠 Features:"
echo "   - 📝 Real Log Analysis with Claude AI"
echo "   - 🔧 Real GitHub Actions Analysis"
echo "   - 📊 Comprehensive RAG Dashboard"
echo "   - 🤖 Smart source identification (RAG vs Claude AI)"
echo "   - 📈 Real-time training progress"
echo ""
echo "🏆 Ready for Hackathon Victory! 🥇"