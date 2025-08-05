# AI Dashboard - Business Analytics Platform

A comprehensive AI-powered business analytics dashboard built with Flask, featuring real-time data visualization, machine learning insights, and interactive AI chat capabilities.

## 🚀 Features
<img width="1316" height="930" alt="Screenshot 2025-08-05 at 1 04 09 PM" src="https://github.com/user-attachments/assets/379deb76-db00-4bc9-975b-c6198668e99d" />

### 📊 **Main Dashboard**
- Real-time business metrics and KPIs
- Interactive charts and visualizations
- AI-powered insights and recommendations
- Responsive design with modern UI

### 🤖 **AI Chat Assistant**
- Natural language business queries
- Contextual responses based on data
- Quick question presets
- Real-time conversation interface

### 📈 **Advanced Analytics**
- Customer segmentation analysis
- Correlation matrix visualization
- Trend analysis and forecasting
- AI-powered insights generation

### 🔮 **Predictions & Forecasting**
- Sales forecasting with confidence metrics
- User growth predictions
- Conversion rate analysis
- AI recommendations for business improvement

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **Python** - Core programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **OpenAI API** - AI chat functionality

### Frontend
- **TailwindCSS** - Styling framework
- **Plotly.js** - Interactive charts
- **jQuery** - DOM manipulation
- **Font Awesome** - Icons

### AI/ML Libraries
- **Transformers** - Natural language processing
- **Sentence Transformers** - Text embeddings
- **ChromaDB** - Vector database
- **Matplotlib/Seaborn** - Data visualization

## 📁 Project Structure

```
ai-dashboard/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html       # Main dashboard
│   ├── ai_chat.html     # AI chat interface
│   ├── analytics.html   # Advanced analytics
│   └── predictions.html # Forecasting page
├── static/              # Static assets
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript files
└── .env                 # Environment variables (create this)
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
cd /Users/kristinacamacho/dev/Projects/learning/AI
git clone <repository-url> ai-dashboard
cd ai-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_secret_key_here
```

### 4. Run the Application
```bash
python app.py
```

The dashboard will be available at `http://localhost:5001`

## 📊 Dashboard Features

### Main Dashboard (`/`)
- **Real-time Statistics**: Sales, users, conversion rates, AI accuracy
- **Interactive Charts**: Line charts, bar charts, scatter plots
- **AI Insights**: Automated trend analysis and predictions
- **Responsive Design**: Works on desktop and mobile devices

### AI Chat (`/ai-chat`)
- **Natural Language Processing**: Ask questions in plain English
- **Business Context**: AI understands business metrics and terminology
- **Quick Actions**: Pre-defined questions for common queries
- **Real-time Responses**: Instant AI-powered insights

### Analytics (`/analytics`)
- **Customer Segmentation**: K-means clustering analysis
- **Correlation Analysis**: Feature relationship visualization
- **Advanced Metrics**: Processing time, data points, AI accuracy
- **Insight Panels**: Trend analysis, behavior patterns, recommendations

### Predictions (`/predictions`)
- **Sales Forecasting**: 30-day and quarterly predictions
- **User Growth**: Predictive user acquisition modeling
- **Confidence Metrics**: AI prediction reliability scores
- **Recommendations**: Actionable business improvement suggestions

## 🔧 API Endpoints

### Data Endpoints
- `GET /api/stats` - Dashboard statistics
- `GET /api/chart/<chart_type>` - Chart data
- `GET /api/insights/<data_type>` - AI insights
- `GET /api/segments` - Customer segmentation

### AI Chat
- `POST /api/chat` - AI chat responses

### Page Routes
- `GET /` - Main dashboard
- `GET /analytics` - Analytics page
- `GET /predictions` - Predictions page
- `GET /ai-chat` - AI chat interface

## 🎯 Key Features

### AI-Powered Insights
- **Trend Analysis**: Automatic pattern recognition
- **Predictive Modeling**: Future performance forecasting
- **Recommendation Engine**: Actionable business advice
- **Confidence Scoring**: Reliability metrics for predictions

### Data Visualization
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Live data refresh
- **Responsive Design**: Mobile-friendly interface
- **Custom Styling**: Modern gradient designs

### Business Intelligence
- **KPI Tracking**: Key performance indicators
- **Customer Segmentation**: Behavioral analysis
- **Correlation Analysis**: Feature relationships
- **Performance Metrics**: AI accuracy and processing time

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: International business data
- **Advanced Document Processing**: PDF/Word document analysis
- **Real-time Data Integration**: Live database connections
- **Advanced ML Models**: Deep learning implementations
- **User Authentication**: Multi-user support
- **Cloud Deployment**: AWS/Azure integration

### Technical Improvements
- **Microservices Architecture**: Scalable backend design
- **Enhanced Security**: Authentication and authorization
- **Performance Optimization**: Caching and optimization
- **Advanced Analytics**: More sophisticated ML models
- **API Documentation**: Comprehensive API docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Kristina Camacho**
- AI/ML Developer
- Full-stack Engineer
- Business Analytics Specialist

## 🆘 Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Built with ❤️ using Flask, Python, and modern AI technologies** 
