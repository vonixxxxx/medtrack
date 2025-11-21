const { PrismaClient } = require('@prisma/client');
require('dotenv').config();

// Create a single PrismaClient instance to be shared across all controllers
const prisma = new PrismaClient();

module.exports = prisma;



