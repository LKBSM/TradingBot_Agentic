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
  created_at: string;
  consents: Consent[];
}

export interface RegisterInput {
  username: string;
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
