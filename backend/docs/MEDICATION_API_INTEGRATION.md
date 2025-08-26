# Enhanced Medication API Integration

## Overview

MedTrack now features a comprehensive medication search and information system that integrates with multiple authoritative drug databases and provides AI-powered summarization of medication information. The system prioritizes the European Medicines Agency (EMA) API while maintaining compatibility with existing sources.

## Features

### 🚀 Enhanced EMA Integration
- **Comprehensive Data**: Fetches detailed medication information including safety data, side effects, and contraindications
- **Real-time Updates**: Connects directly to EMA's live API endpoints
- **Rich Metadata**: Includes therapeutic indications, special population guidance, and authorization status

### 🤖 AI-Powered Summarization
- **OpenAI Integration**: Uses GPT-3.5-turbo for intelligent medication summaries
- **HuggingFace Fallback**: Alternative LLM integration for cost-effective summarization
- **Rule-based Fallback**: Intelligent fallback when LLM services are unavailable
- **Patient-Friendly**: Converts complex medical information into clear, actionable summaries

### 🔍 Multi-Source Search
- **EMA (Enhanced)**: Primary source with comprehensive European medication data
- **RxNorm**: US National Library of Medicine canonical drug identifiers
- **openFDA**: US Food and Drug Administration safety and labeling information
- **NHS dm+d**: UK National Health Service drug dictionary
- **Unified Results**: Intelligent deduplication and relevance scoring across sources

### 🛡️ Safety & Compliance
- **Privacy-First**: No PHI sent to external APIs
- **Secure Caching**: In-memory caching with configurable TTL
- **Rate Limiting**: Built-in protection against API abuse
- **Error Handling**: Graceful degradation when external services fail

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │  External APIs  │
│                 │    │                  │    │                 │
│ MedicationLibrary│◄──►│ Medication       │◄──►│ EMA API         │
│ Component       │    │ Controller       │    │ RxNorm API      │
│                 │    │                  │    │ openFDA API     │
│ Enhanced Modal  │    │ Enhanced         │    │ NHS dm+d API    │
│ with AI Summary │    │ Medication       │    │                 │
└─────────────────┘    │ Service          │    └─────────────────┘
                       │                  │
                       │ LLM              │    ┌─────────────────┐
                       │ Summarization    │◄──►│ OpenAI API      │
                       │                  │    │ HuggingFace API │
                       │ Caching Layer    │    └─────────────────┘
                       └──────────────────┘
```

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the backend directory:

```bash
# Required
DATABASE_URL="file:./dev.db"
JWT_SECRET="your-super-secure-secret-key"
PORT=8000

# Optional - LLM API Keys
OPENAI_API_KEY="your-openai-api-key"
HUGGINGFACE_API_KEY="your-huggingface-api-key"

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

### 2. Install Dependencies

```bash
cd backend
npm install openai @huggingface/inference
```

### 3. API Key Setup

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and add billing information
3. Generate an API key
4. Add to your `.env` file

#### HuggingFace API Key (Alternative)
1. Visit [HuggingFace](https://huggingface.co/)
2. Create an account
3. Go to Settings → Access Tokens
4. Generate a new token
5. Add to your `.env` file

### 4. Start the Backend

```bash
npm run dev
```

## API Endpoints

### Search Medications
```http
GET /api/medications/search?query={query}&limit={limit}&source={source}
```

**Parameters:**
- `query` (required): Search term (minimum 2 characters)
- `limit` (optional): Maximum results (default: 20)
- `source` (optional): Data source filter (`ema`, `all`, `rxnorm`, `openfda`, `nhs_dmd`)

**Example:**
```bash
curl "http://localhost:8000/api/medications/search?query=aspirin&limit=10&source=ema"
```

**Response:**
```json
{
  "data": [
    {
      "source": "ema",
      "id": "EPAR-12345",
      "name": "Aspirin",
      "genericName": "Acetylsalicylic acid",
      "brandNames": ["Aspirin", "Bayer Aspirin"],
      "dosageForms": ["Tablet", "Oral solution"],
      "strengths": ["100mg", "300mg", "500mg"],
      "atcClass": "B01AC06",
      "therapeuticIndications": ["Pain relief", "Fever reduction"],
      "warnings": ["May cause stomach irritation"],
      "sideEffects": ["Nausea", "Stomach upset"],
      "interactions": ["Blood thinners", "NSAIDs"],
      "pregnancyCategory": "Category D",
      "breastfeedingCategory": "Use with caution",
      "pediatricUse": "Not recommended under 16",
      "geriatricUse": "Use with caution",
      "llmSummary": "AI-generated patient-friendly summary..."
    }
  ],
  "cached": false,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "total": 1,
  "query": "aspirin",
  "sources": ["ema"]
}
```

### Get Medication Details
```http
GET /api/medications/details/{source}/{id}
```

**Parameters:**
- `source`: Data source identifier
- `id`: Medication identifier

**Example:**
```bash
curl "http://localhost:8000/api/medications/details/ema/EPAR-12345"
```

### Get Data Sources
```http
GET /api/medications/sources
```

**Response:**
```json
{
  "sources": [
    {
      "id": "ema",
      "name": "EMA Medicines Data",
      "description": "European Medicines Agency product registry and authorization data",
      "baseUrl": "https://www.ema.europa.eu",
      "features": ["EU medicines", "Authorization status", "Product information", "ATC codes"]
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "total": 1
}
```

## LLM Integration

### OpenAI Integration
The system automatically uses OpenAI's GPT-3.5-turbo model when an API key is provided:

```javascript
const summary = await MedicationSummarizer.summarizeWithOpenAI(medicationData);
```

**Features:**
- Medical context awareness
- Patient-friendly language
- Structured output formatting
- Safety information prioritization

### HuggingFace Integration
Fallback to HuggingFace models for cost-effective summarization:

```javascript
const summary = await MedicationSummarizer.summarizeWithHuggingFace(medicationData);
```

**Models Used:**
- `microsoft/DialoGPT-medium` for text generation
- Configurable parameters for quality vs. speed

### Fallback Summarization
When LLM services are unavailable, the system provides intelligent rule-based summaries:

```javascript
const summary = MedicationSummarizer.fallbackSummarization(medicationData);
```

**Features:**
- Structured information extraction
- Priority-based organization
- Clear categorization of safety information

## Frontend Integration

### Enhanced Medication Library Component

The frontend component automatically detects and displays:

1. **AI-Powered Summary**: Prominently displayed at the top
2. **Safety Information**: Warnings and contraindications in red-highlighted sections
3. **Side Effects**: Organized by severity with visual indicators
4. **Special Populations**: Pregnancy, breastfeeding, pediatric, and geriatric guidance
5. **Therapeutic Information**: Indications, dosage forms, and strengths
6. **Source Attribution**: Clear indication of data origin

### Search Functionality

```javascript
const handleSearch = async (query) => {
  const params = new URLSearchParams({
    query: query.trim(),
    limit: 20,
    source: activeFilter // 'ema', 'all', 'rxnorm', 'openfda', 'nhs_dmd'
  });

  const response = await fetch(`${API_BASE}/api/medications/search?${params}`);
  const data = await response.json();
  setSearchResults(data.data || []);
};
```

### Modal Display

Enhanced medication details modal with:

- **Collapsible Sections**: Organized information display
- **Visual Indicators**: Color-coded safety information
- **Interactive Elements**: Expandable details and source links
- **Responsive Design**: Mobile-friendly layout

## Caching Strategy

### In-Memory Caching
- **Search Results**: 30-minute TTL for frequently accessed queries
- **Drug Details**: 2-hour TTL for comprehensive medication information
- **LLM Summaries**: Cached with medication details to avoid regeneration

### Cache Management
```javascript
// Get cache statistics
GET /api/medications/cache/stats

// Clear cache (admin only)
DELETE /api/medications/cache
```

## Error Handling

### Graceful Degradation
1. **Primary Source Failure**: Falls back to alternative sources
2. **LLM Service Unavailable**: Uses rule-based summarization
3. **Network Issues**: Returns cached results when possible
4. **API Rate Limits**: Implements exponential backoff

### Error Responses
```json
{
  "error": "Failed to search medications",
  "details": "EMA API rate limit exceeded",
  "fallback": "Using cached results from previous search"
}
```

## Testing

### Unit Tests
Run the comprehensive test suite:

```bash
cd backend
npm test tests/medicationService.test.js
```

### Test Coverage
- API integration testing
- LLM summarization testing
- Error handling scenarios
- Performance and caching tests
- Edge case handling

## Performance Considerations

### Optimization Strategies
1. **Parallel API Calls**: Simultaneous requests to multiple sources
2. **Intelligent Caching**: TTL-based cache invalidation
3. **Request Batching**: Grouped API calls for efficiency
4. **Connection Pooling**: Reuse HTTP connections

### Monitoring
- API response times
- Cache hit rates
- LLM summarization latency
- Error rates by source

## Security & Privacy

### Data Protection
- **No PHI Transmission**: Personal health information never sent to external APIs
- **Secure Caching**: In-memory storage only, no persistent storage of sensitive data
- **API Key Security**: Environment variable storage, never logged or exposed
- **Rate Limiting**: Protection against abuse and DoS attacks

### Compliance
- **GDPR Ready**: European data protection compliance
- **HIPAA Compatible**: US healthcare privacy standards
- **Audit Logging**: Track all external API interactions
- **Data Minimization**: Only request necessary information

## Troubleshooting

### Common Issues

#### EMA API Errors
```bash
# Check API connectivity
curl "https://www.ema.europa.eu/en/medicines/api/medicines?search=aspirin&limit=1"

# Verify rate limits
# EMA has strict rate limiting - implement exponential backoff
```

#### LLM Service Issues
```bash
# Test OpenAI connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     "https://api.openai.com/v1/models"

# Test HuggingFace connectivity
curl -H "Authorization: Bearer $HUGGINGFACE_API_KEY" \
     "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
```

#### Performance Issues
```bash
# Check cache statistics
curl "http://localhost:8000/api/medications/cache/stats"

# Monitor memory usage
# Large result sets may require cache size adjustment
```

### Debug Mode
Enable detailed logging:

```javascript
// In medicationService.js
const DEBUG = process.env.NODE_ENV === 'development';
if (DEBUG) {
  console.log('API Request:', { url, params });
  console.log('API Response:', response.data);
}
```

## Future Enhancements

### Planned Features
1. **Offline Mode**: Local medication database for offline access
2. **Advanced Filtering**: Therapeutic class, contraindication-based filtering
3. **Drug Interaction Checking**: Real-time interaction validation
4. **Personalized Summaries**: User preference-based information prioritization
5. **Multi-language Support**: Localized medication information

### API Expansions
1. **WHO Drug Information**: Global medication database
2. **Clinical Trials Data**: Research and development information
3. **Pharmacovigilance**: Adverse event reporting integration
4. **Cost Information**: Pricing and availability data

## Support & Contributing

### Getting Help
1. Check the troubleshooting section above
2. Review API documentation for external services
3. Examine test cases for usage examples
4. Check GitHub issues for known problems

### Contributing
1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

### License
This medication integration system is part of MedTrack and follows the same licensing terms.

---

**Note**: This system is designed for educational and informational purposes. Always consult healthcare professionals for medical advice. The AI-generated summaries should not replace professional medical consultation.
