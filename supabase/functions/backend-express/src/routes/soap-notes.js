const express = require('express');
const router = express.Router();
const {
  getSoapNotes,
  getSoapNote,
  createSoapNote,
  updateSoapNote,
  deleteSoapNote
} = require('../controllers/soapNoteController');

router.get('/', getSoapNotes);
router.get('/:id', getSoapNote);
router.post('/', createSoapNote);
router.put('/:id', updateSoapNote);
router.delete('/:id', deleteSoapNote);

module.exports = router;



