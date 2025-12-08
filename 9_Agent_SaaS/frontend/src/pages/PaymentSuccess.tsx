import { useEffect, useState } from 'react';
import { useStripe } from '@stripe/react-stripe-js';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/lib/supabase';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Loader2, Coins } from 'lucide-react';

export default function PaymentSuccess() {
  const stripe = useStripe();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [message, setMessage] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [tokenBalance, setTokenBalance] = useState<number | null>(null);
  const [updatingBalance, setUpdatingBalance] = useState(false);

  useEffect(() => {
    const fetchTokenBalance = async () => {
      if (!user?.id) return;

      try {
        const { data, error } = await supabase
          .from('user_profiles')
          .select('tokens')
          .eq('id', user.id)
          .single();

        if (error) throw error;
        setTokenBalance(data?.tokens || 0);
      } catch (error) {
        console.error('Error fetching token balance:', error);
      }
    };

    const pollForTokenUpdate = async (initialBalance: number) => {
      setUpdatingBalance(true);

      // Check immediately first (tokens might already be there)
      try {
        const { data, error } = await supabase
          .from('user_profiles')
          .select('tokens')
          .eq('id', user.id)
          .single();

        if (error) throw error;

        const newBalance = data?.tokens || 0;
        if (newBalance > initialBalance) {
          setTokenBalance(newBalance);
          setUpdatingBalance(false);
          return;
        }
      } catch (error) {
        console.error('Error fetching token balance:', error);
      }

      // If tokens aren't there yet, set up real-time subscription
      const channel = supabase
        .channel('payment-token-updates')
        .on(
          'postgres_changes',
          {
            event: 'UPDATE',
            schema: 'public',
            table: 'user_profiles',
            filter: `id=eq.${user.id}`,
          },
          (payload) => {
            if (payload.new && 'tokens' in payload.new) {
              const newTokens = (payload.new as { tokens: number }).tokens;
              if (newTokens > initialBalance) {
                setTokenBalance(newTokens);
                setUpdatingBalance(false);
                supabase.removeChannel(channel);
              }
            }
          }
        )
        .subscribe();

      // Safety timeout - stop waiting after 3 seconds
      setTimeout(() => {
        setUpdatingBalance(false);
        fetchTokenBalance();
        supabase.removeChannel(channel);
      }, 3000);
    };

    if (!stripe || !user?.id) return;

    const clientSecret = new URLSearchParams(window.location.search).get(
      'payment_intent_client_secret'
    );

    if (!clientSecret) {
      setMessage('No payment information found');
      setLoading(false);
      return;
    }

    stripe.retrievePaymentIntent(clientSecret).then(async ({ paymentIntent }) => {
      if (!paymentIntent) {
        setMessage('Payment information not found');
        setLoading(false);
        return;
      }

      switch (paymentIntent.status) {
        case 'succeeded':
          setMessage('Payment succeeded! Your tokens have been added to your account.');

          const { data } = await supabase
            .from('user_profiles')
            .select('tokens')
            .eq('id', user.id)
            .single();

          const currentBalance = data?.tokens || 0;
          setTokenBalance(currentBalance);

          pollForTokenUpdate(currentBalance);
          break;
        case 'processing':
          setMessage('Your payment is processing. Tokens will be added shortly.');
          break;
        case 'requires_payment_method':
          setMessage('Your payment was not successful, please try again.');
          break;
        default:
          setMessage('Something went wrong.');
          break;
      }
      setLoading(false);
    });
  }, [stripe, user?.id]);

  return (
    <div className="container mx-auto p-6 max-w-2xl">
      <Card className="mt-12">
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            {loading ? (
              <Loader2 className="h-16 w-16 text-primary animate-spin" />
            ) : (
              <CheckCircle2 className="h-16 w-16 text-green-500" />
            )}
          </div>
          <CardTitle className="text-center text-3xl">
            {loading ? 'Processing...' : 'Payment Complete'}
          </CardTitle>
          <CardDescription className="text-center text-lg">
            {loading ? 'Verifying your payment' : 'Thank you for your purchase'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {message && !loading && (
            <Alert variant={message.includes('succeeded') ? 'default' : 'destructive'}>
              <AlertDescription className="text-center">{message}</AlertDescription>
            </Alert>
          )}
          {tokenBalance !== null && (
            <div className="flex items-center justify-center gap-2 p-4 bg-muted rounded-lg">
              <Coins className="h-6 w-6 text-primary" />
              <span className="text-lg font-medium">
                {updatingBalance ? 'Updating Balance...' : 'New Balance:'}
              </span>
              {updatingBalance ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Badge variant="secondary" className="text-lg px-4 py-2">
                  {tokenBalance} tokens
                </Badge>
              )}
            </div>
          )}
        </CardContent>
        <CardFooter className="flex justify-center gap-4">
          <Button onClick={() => navigate('/')} size="lg">
            Back to Chat
          </Button>
          <Button onClick={() => navigate('/token-history')} variant="outline" size="lg">
            View History
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}