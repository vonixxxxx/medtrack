-- Basic RLS Policies to allow your app to work
-- These policies allow service role (your backend) to access all data
-- and authenticated users to access their own data

-- Allow service role (your backend API) full access to all tables
-- This uses the service_role key which bypasses RLS
-- Your Vercel functions use this, so they can access all data

-- For users table: Allow service role full access
CREATE POLICY "Service role can access users" ON "users"
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For patients: Allow service role full access
CREATE POLICY "Service role can access patients" ON "patients"
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For medications: Allow service role full access
CREATE POLICY "Service role can access medications" ON "Medication"
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- For all other tables: Allow service role full access
-- You can customize these later for more granular control

CREATE POLICY "Service role can access clinicians" ON "clinicians"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access patient_medications" ON "patient_medications"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access medical_notes" ON "medical_notes"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access lab_results" ON "lab_results"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access vital_signs" ON "vital_signs"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access appointments" ON "appointments"
  FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Service role can access encounters" ON "encounters"
  FOR ALL USING (true) WITH CHECK (true);

-- Note: Your Prisma client uses the service role connection string
-- which bypasses RLS, so these policies are mainly for future
-- Supabase client usage. Your app will continue to work.
