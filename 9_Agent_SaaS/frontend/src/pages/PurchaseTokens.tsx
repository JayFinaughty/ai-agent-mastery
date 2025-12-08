import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { loadStripe } from '@stripe/stripe-js';
import { Elements } from '@stripe/react-stripe-js';
import { useAuth } from '@/hooks/useAuth';
import CheckoutForm from '@/components/tokens/CheckoutForm';
import PricingCard from '@/components/tokens/PricingCard';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';

const stripePromise = loadStripe(
  import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY || ''
);

const PRICING_TIERS = [
  { id: 'tier_1', name: '100 Tokens', price: 5, tokens: 100 },
  { id: 'tier_2', name: '250 Tokens', price: 10, tokens: 250 },
  { id: 'tier_3', name: '600 Tokens', price: 20, tokens: 600 },
];

export default function PurchaseTokens() {
  const { user, session } = useAuth();
  const navigate = useNavigate();
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTierSelect = async (tierId: string) => {
    setSelectedTier(tierId);
    setLoading(true);
    setError(null);

    try {
      const agentEndpoint = import.meta.env.VITE_AGENT_ENDPOINT || '';
      const baseUrl = agentEndpoint.replace('/api/pydantic-agent', '');

      const response = await fetch(`${baseUrl}/api/create-payment-intent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({
          tier: tierId,
          user_id: user?.id,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create payment intent');
      }

      const data = await response.json();
      setClientSecret(data.client_secret);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setSelectedTier(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Back Button */}
      <div className="mb-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => clientSecret ? setClientSecret(null) : navigate('/')}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          {clientSecret ? 'Back to Packages' : 'Back to Chat'}
        </Button>
      </div>

      {/* Centered Header */}
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold mb-3">Purchase Tokens</h1>
        <p className="text-muted-foreground text-lg mb-3">
          Select a token package to continue using the AI agent
        </p>
        <Button
          variant="link"
          size="sm"
          onClick={() => navigate('/token-history')}
          className="text-sm"
        >
          View Transaction History →
        </Button>
      </div>

      {!clientSecret && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {PRICING_TIERS.map((tier) => (
            <PricingCard
              key={tier.id}
              tier={tier.id}
              name={tier.name}
              price={tier.price}
              tokens={tier.tokens}
              selected={selectedTier === tier.id}
              onClick={() => handleTierSelect(tier.id)}
              loading={loading && selectedTier === tier.id}
            />
          ))}
        </div>
      )}

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {clientSecret && (
        <div className="max-w-2xl mx-auto">
          <Elements stripe={stripePromise} options={{ clientSecret }}>
            <CheckoutForm />
          </Elements>
        </div>
      )}
    </div>
  );
}