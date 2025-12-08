import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/lib/supabase';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Coins } from 'lucide-react';

export default function TokenBalance() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [tokens, setTokens] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user?.id) return;

    const fetchTokens = async () => {
      try {
        const { data, error } = await supabase
          .from('user_profiles')
          .select('tokens')
          .eq('id', user.id)
          .single();

        if (error) throw error;
        setTokens(data?.tokens || 0);
      } catch (error) {
        console.error('Error fetching tokens:', error);
        setTokens(0);
      } finally {
        setLoading(false);
      }
    };

    fetchTokens();

    const channel = supabase
      .channel('token-updates')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'user_profiles',
          filter: `id=eq.${user.id}`,
        },
        (payload) => {
          if (payload.new && 'tokens' in payload.new) {
            setTokens((payload.new as { tokens: number }).tokens);
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [user?.id]);

  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <Coins className="h-5 w-5" />
        <span>Loading...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <Coins className="h-5 w-5 text-yellow-500" />
        <span className="text-base font-medium text-white">
          {tokens} {tokens === 1 ? 'token' : 'tokens'}
        </span>
      </div>
      <Button
        onClick={() => navigate('/purchase-tokens')}
        size="sm"
        variant={tokens < 10 ? 'default' : 'outline'}
      >
        {tokens < 10 ? 'Buy Tokens' : 'Add More'}
      </Button>
    </div>
  );
}