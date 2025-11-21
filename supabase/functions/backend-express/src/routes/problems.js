const express = require('express');
const router = express.Router();
const {
  getProblems,
  getProblem,
  createProblem,
  updateProblem,
  deleteProblem
} = require('../controllers/problemController');

router.get('/', getProblems);
router.get('/:id', getProblem);
router.post('/', createProblem);
router.put('/:id', updateProblem);
router.delete('/:id', deleteProblem);

module.exports = router;



