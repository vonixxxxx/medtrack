# Global Drug Data API Integration

This document describes the integration of authoritative global drug data APIs into the MedTrack application.

## Overview

The Medications page now integrates with real-time APIs from multiple authoritative sources:
- **RxNorm (US NLM)**: Canonical drug IDs and synonyms
- **openFDA**: US drug labeling and safety information
- **NHS dm+d**: UK drug dictionary and NHS medicines
- **EMA**: European Medicines Agency product registry

## Key Features

✅ **Real-time Data**: No hardcoded medications - all data fetched live from APIs
✅ **Multi-source Search**: Query all sources simultaneously or filter by specific source
✅ **Data Normalization**: Unified schema across different data sources
✅ **Intelligent Caching**: Short-term in-memory caching for performance
✅ **Source Attribution**: Clear indication of data source and last update time
✅ **Professional UI**: Enterprise-grade interface with source-specific styling

## API Endpoints

### 1. Search Medications
```
GET /api/medications/search?query={search_term}&limit={number}&source={source}
```

**Parameters:**
- `query` (required): Search term (minimum 2 characters)
- `limit` (optional): Maximum results (default: 20)
- `source` (optional): Specific data source (rxnorm, openfda, nhs_dmd, ema)

**Example:**
```bash
curl "http://localhost:3001/api/medications/search?query=aspirin&limit=10"
```

**Response:**
```json
{
  "data": [
    {
      "id": "1191",
      "name": "Aspirin",
      "genericName": "Acetylsalicylic acid",
      "brandNames": ["Bayer Aspirin", "Ecotrin"],
      "dosageForms": ["Tablet", "Chewable tablet"],
      "strengths": ["81mg", "325mg"],
      "atcClass": "B01AC06",
      "source": "rxnorm",
      "sourceUrl": "https://rxnav.nlm.nih.gov/REST/rxcui/1191/allrelated.json",
      "identifiers": {
        "rxcui": "1191",
        "ndc": ["00071-1015", "00071-1016"]
      },
      "usData": {
        "fdaApproved": true,
        "ndcCodes": ["00071-1015", "00071-1016"],
        "rxcui": "1191"
      },
      "metadata": {
        "source": "rxnorm",
        "lastUpdated": "2024-01-15T10:30:00.000Z"
      }
    }
  ],
  "cached": false,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "total": 1,
  "query": "aspirin",
  "sources": ["rxnorm", "openfda", "nhs_dmd", "ema"]
}
```

### 2. Get Medication Details
```
GET /api/medications/details/{source}/{id}
```

**Parameters:**
- `source`: Data source identifier
- `id`: Medication ID from the source

**Example:**
```bash
curl "http://localhost:3001/api/medications/details/rxnorm/1191"
```

### 3. Get Data Sources
```
GET /api/medications/sources
```

**Response:**
```json
{
  "sources": [
    {
      "id": "rxnorm",
      "name": "RxNorm (US NLM)",
      "description": "Canonical drug IDs and synonyms from US National Library of Medicine",
      "baseUrl": "https://rxnav.nlm.nih.gov",
      "features": ["Drug identification", "Synonyms", "ATC codes", "NDC mapping"]
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "total": 4
}
```

### 4. Cache Management
```
GET /api/medications/cache/stats
DELETE /api/medications/cache
```

## Data Sources

### RxNorm (US NLM)
- **Base URL**: https://rxnav.nlm.nih.gov/REST
- **Purpose**: Canonical drug identification and synonyms
- **Data**: Drug names, RxCUI codes, NDC mapping, ATC classification
- **Rate Limits**: None specified, but use reasonable request frequency

### openFDA
- **Base URL**: https://api.fda.gov
- **Purpose**: US drug labeling and safety information
- **Data**: Drug labels, NDC codes, warnings, adverse reactions, interactions
- **Rate Limits**: 1000 requests per day, 1000 requests per hour

### NHS dm+d
- **Base URL**: https://directory.spineservices.nhs.uk/ORD/2-0-0
- **Purpose**: UK drug dictionary and NHS medicines
- **Data**: UK medicines, dm+d codes, VMP/AMP mapping
- **Rate Limits**: None specified

### EMA
- **Base URL**: https://www.ema.europa.eu/en/medicines/api
- **Purpose**: European Medicines Agency product registry
- **Data**: EU medicines, authorization status, product information
- **Rate Limits**: None specified

## Data Normalization

All drug data is normalized into a consistent schema:

```typescript
interface NormalizedDrug {
  id: string;
  name: string;
  genericName: string;
  brandNames: string[];
  dosageForms: string[];
  strengths: string[];
  atcClass: string;
  source: string;
  sourceUrl: string;
  identifiers: Record<string, any>;
  metadata: {
    source: string;
    lastUpdated: string;
  };
  usData?: USDrugData;
  ukData?: UKDrugData;
  euData?: EUDrugData;
  safety?: SafetyData;
}
```

## Caching Strategy

- **Search Results**: 30 minutes TTL
- **Drug Details**: 2 hours TTL
- **Maximum Cache Size**: 1000 keys
- **Automatic Cleanup**: Every 5 minutes
- **No Persistent Storage**: All cache is in-memory only

## Error Handling

The API gracefully handles:
- Network timeouts (10 second limit)
- API rate limiting
- Invalid responses
- Missing data
- Source unavailability

## Frontend Integration

The frontend automatically:
- Loads data sources on page load
- Implements real-time search with debouncing
- Displays source-specific information
- Shows loading states and error messages
- Provides source filtering options

## Security Considerations

- **Public Endpoints**: Drug search and details are publicly accessible
- **Rate Limiting**: Implemented at the application level
- **Input Validation**: All search queries are sanitized
- **CORS**: Configured for frontend access
- **No Sensitive Data**: Only public drug information is exposed

## Performance Optimization

- **Parallel API Calls**: Multiple sources queried simultaneously
- **Intelligent Caching**: Reduces API calls for repeated searches
- **Request Deduplication**: Prevents duplicate API calls
- **Timeout Management**: Prevents hanging requests
- **Error Recovery**: Continues search even if some sources fail

## Monitoring and Debugging

### Cache Statistics
```bash
curl "http://localhost:3001/api/medications/cache/stats"
```

### Health Check
The cache provides health metrics including:
- Hit rate percentage
- Total requests
- Memory usage
- Cache size

## Troubleshooting

### Common Issues

1. **API Timeouts**
   - Check network connectivity
   - Verify API endpoints are accessible
   - Review timeout configuration (currently 10 seconds)

2. **Rate Limiting**
   - Monitor openFDA usage (1000 requests/day)
   - Implement exponential backoff for failed requests
   - Use caching to reduce API calls

3. **Data Inconsistencies**
   - Different sources may have different data
   - Normalization handles most cases
   - Source-specific data is preserved

### Debug Mode

Enable detailed logging by setting:
```bash
DEBUG=drug-api:*
```

## Future Enhancements

- **DrugBank Integration**: Clinical monographs and interactions (requires API key)
- **Real-time Updates**: WebSocket connections for live data
- **Advanced Filtering**: By therapeutic area, approval status, etc.
- **Bulk Export**: CSV/JSON export of search results
- **User Preferences**: Save frequently searched medications

## Compliance

This integration complies with:
- **HIPAA**: No patient data is stored or transmitted
- **GDPR**: Only public drug information is accessed
- **API Terms**: Respects rate limits and usage policies
- **Data Privacy**: No personal information is collected

## Support

For technical support or questions about the API integration:
1. Check the application logs for error details
2. Verify API endpoints are accessible
3. Review rate limiting and timeout settings
4. Check cache statistics for performance issues
