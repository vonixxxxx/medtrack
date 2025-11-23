const express = require('express');
const router = express.Router();
const {
  getCharges,
  createCharge,
  updateCharge,
  getPayments,
  createPayment
} = require('../controllers/billingController');

router.get('/charges', getCharges);
router.post('/charges', createCharge);
router.put('/charges/:id', updateCharge);

router.get('/payments', getPayments);
router.post('/payments', createPayment);

module.exports = router;



