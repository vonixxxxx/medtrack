const nock = require('nock');
const { validateMedication } = require('../src/services/medicationMatchingService');

function mockBioGPTSuccess(drugName) {
  return async (prompt) => {
    return {
      is_medication: true,
      drug_class: 'ACE inhibitor',
      is_generic: true,
      is_brand: false,
      confidence: 0.95
    };
  };
}

function mockBioGPTReject() {
  return async (prompt) => ({
    is_medication: false,
    drug_class: null,
    is_generic: false,
    is_brand: false,
    confidence: 0.1
  });
}

afterEach(() => {
  nock.cleanAll();
});

test('reject greeting "hello"', async () => {
  const r = await validateMedication('hello', { callBioGPTFn: mockBioGPTSuccess('') });
  expect(r.found).toBe(false);
  expect(r.reason).toBe('greeting');
});

test('fuzzy misspelling "paracetmol" -> paracetamol (RxNorm verified)', async () => {
  // mock RxNorm approximateTerm
  nock('https://rxnav.nlm.nih.gov')
    .get(/\/REST\/approximateTerm.json/)
    .reply(200, {
      approximateGroup: {
        candidate: [
          { rxcui: '12345', score: 95, rank: '1', term: 'paracetamol' }
        ]
      }
    });

  const r = await validateMedication('paracetmol', { callBioGPTFn: mockBioGPTSuccess('paracetamol') });
  expect(r.found).toBe(true);
  expect(r.name).toBe('paracetamol');
  expect(r.source).toBe('rxnorm');
  expect(r.score).toBeCloseTo(0.95, 2);
  expect(r.bio.confidence).toBeGreaterThan(0.7);
});

test('brand "xanax" maps to alprazolam (rxnorm + biogpt)', async () => {
  nock('https://rxnav.nlm.nih.gov')
    .get(/\/REST\/approximateTerm.json/)
    .reply(200, {
      approximateGroup: {
        candidate: [ { rxcui: '22222', score: 92, rank: '1', term: 'alprazolam' } ]
      }
    });

  const r = await validateMedication('xanax', { callBioGPTFn: mockBioGPTSuccess('alprazolam') });
  expect(r.found).toBe(true);
  expect(r.name).toBe('alprazolam');
  expect(r.rxcui).toBe('22222');
});

test('no authoritative match should return suggestions', async () => {
  // RxNorm returns nothing
  nock('https://rxnav.nlm.nih.gov')
    .get(/\/REST\/approximateTerm.json/)
    .reply(200, { approximateGroup: {} });

  const r = await validateMedication('someweirdword', { callBioGPTFn: mockBioGPTSuccess('') });
  expect(r.found).toBe(false);
  expect(r.reason).toBe('no_authoritative_match');
  expect(Array.isArray(r.suggestions)).toBe(true);
});


