const express = require('express');
const router = express.Router();
const {
  getDiaryEntries,
  createDiaryEntry,
  updateDiaryEntry,
  deleteDiaryEntry
} = require('../controllers/diaryController');

router.get('/', getDiaryEntries);
router.post('/', createDiaryEntry);
router.put('/:id', updateDiaryEntry);
router.delete('/:id', deleteDiaryEntry);

module.exports = router;



