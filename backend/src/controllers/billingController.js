const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const getCharges = async (req, res) => {
  try {
    const { patientId, encounterId, status } = req.query;
    const where = {};

    if (patientId) where.patientId = patientId;
    if (encounterId) where.encounterId = encounterId;
    if (status) where.status = status;

    const charges = await prisma.charge.findMany({
      where,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        encounter: true
      },
      orderBy: { dateOfService: 'desc' }
    });

    res.json(charges);
  } catch (error) {
    console.error('Error fetching charges:', error);
    res.status(500).json({ error: 'Failed to fetch charges' });
  }
};

const createCharge = async (req, res) => {
  try {
    const {
      patientId,
      encounterId,
      code,
      codeType,
      description,
      units,
      fee,
      dateOfService,
      providerId
    } = req.body;

    if (!patientId || !code || !fee || !dateOfService) {
      return res.status(400).json({ error: 'Patient ID, code, fee, and date of service are required' });
    }

    const charge = await prisma.charge.create({
      data: {
        patientId,
        encounterId: encounterId || null,
        code,
        codeType: codeType || 'CPT',
        description: description || null,
        units: units || 1,
        fee,
        dateOfService: new Date(dateOfService),
        providerId: providerId || null,
        status: 'pending',
        createdBy: req.user?.id || null
      },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        }
      }
    });

    res.status(201).json(charge);
  } catch (error) {
    console.error('Error creating charge:', error);
    res.status(500).json({ error: 'Failed to create charge' });
  }
};

const updateCharge = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = {};

    const fields = ['code', 'codeType', 'description', 'units', 'fee', 'dateOfService', 'status'];
    fields.forEach(field => {
      if (req.body[field] !== undefined) {
        if (field === 'dateOfService') {
          updateData[field] = new Date(req.body[field]);
        } else {
          updateData[field] = req.body[field];
        }
      }
    });

    const charge = await prisma.charge.update({
      where: { id },
      data: updateData
    });

    res.json(charge);
  } catch (error) {
    console.error('Error updating charge:', error);
    res.status(500).json({ error: 'Failed to update charge' });
  }
};

const getPayments = async (req, res) => {
  try {
    const { patientId } = req.query;
    const where = patientId ? { patientId } : {};

    const payments = await prisma.payment.findMany({
      where,
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        allocations: {
          include: {
            charge: true
          }
        }
      },
      orderBy: { paymentDate: 'desc' }
    });

    res.json(payments);
  } catch (error) {
    console.error('Error fetching payments:', error);
    res.status(500).json({ error: 'Failed to fetch payments' });
  }
};

const createPayment = async (req, res) => {
  try {
    const {
      patientId,
      encounterId,
      amount,
      paymentMethod,
      paymentDate,
      checkNumber,
      creditCardLast4,
      notes,
      allocations
    } = req.body;

    if (!patientId || !amount || !paymentDate) {
      return res.status(400).json({ error: 'Patient ID, amount, and payment date are required' });
    }

    const payment = await prisma.payment.create({
      data: {
        patientId,
        encounterId: encounterId || null,
        amount,
        paymentMethod: paymentMethod || null,
        paymentDate: new Date(paymentDate),
        checkNumber: checkNumber || null,
        creditCardLast4: creditCardLast4 || null,
        notes: notes || null,
        createdBy: req.user?.id || null,
        allocations: allocations ? {
          create: allocations.map(allocation => ({
            chargeId: allocation.chargeId,
            amount: allocation.amount
          }))
        } : undefined
      },
      include: {
        patient: {
          include: {
            user: { select: { id: true, name: true, email: true } }
          }
        },
        allocations: {
          include: {
            charge: true
          }
        }
      }
    });

    res.status(201).json(payment);
  } catch (error) {
    console.error('Error creating payment:', error);
    res.status(500).json({ error: 'Failed to create payment' });
  }
};

module.exports = {
  getCharges,
  createCharge,
  updateCharge,
  getPayments,
  createPayment
};



