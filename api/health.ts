import { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({
    status: 'OK',
    service: 'medtrack-backend',
    timestamp: new Date().toISOString(),
  });
}
