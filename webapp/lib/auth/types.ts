/** Shared account/auth types — mirror of the FastAPI `AccountOut` schema. */

export interface Consent {
  doc: string;
  version: string;
  accepted_at: string;
}

export interface Account {
  id: number;
  username: string;
  email: string;
  role: 'user' | 'owner';
  age_confirmed: boolean;
  /** PAY-1: mandatory email verification before access. */
  email_verified: boolean;
  created_at: string;
  consents: Consent[];
}

export interface RegisterInput {
  // PAY-3: no username — identity is the email; the server derives a username.
  email: string;
  password: string;
  age_confirmed: boolean;
  accept_terms: boolean;
  accept_privacy: boolean;
}

export interface LoginInput {
  identifier: string;
  password: string;
}
