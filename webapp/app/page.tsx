import { redirect } from 'next/navigation';
import { DEFAULT_LOCALE } from '../i18n';

// Root path → redirect to default locale.
export default function RootIndex() {
  redirect(`/${DEFAULT_LOCALE}`);
}
