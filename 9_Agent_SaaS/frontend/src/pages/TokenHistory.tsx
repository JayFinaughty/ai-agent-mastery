import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/lib/supabase';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowDownCircle, ArrowUpCircle, ArrowLeft } from 'lucide-react';

interface Transaction {
  id: string;
  transaction_type: 'purchase' | 'consumption';
  amount: number;
  tokens: number;
  stripe_payment_intent_id: string | null;
  created_at: string;
}

export default function TokenHistory() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user?.id) return;

    const fetchTransactions = async () => {
      try {
        const { data, error } = await supabase
          .from('transactions')
          .select('*')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false })
          .limit(50);

        if (error) throw error;
        setTransactions(data || []);
      } catch (error) {
        console.error('Error fetching transactions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTransactions();
  }, [user?.id]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Back Button */}
      <div className="mb-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/')}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Chat
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Token History</CardTitle>
          <CardDescription>
            View all your token purchases and consumption
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8">Loading...</div>
          ) : transactions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No transactions yet
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Type</TableHead>
                  <TableHead>Tokens</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {transactions.map((transaction) => (
                  <TableRow key={transaction.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {transaction.transaction_type === 'purchase' ? (
                          <>
                            <ArrowUpCircle className="h-4 w-4 text-green-500" />
                            <Badge variant="secondary">
                              Purchase
                            </Badge>
                          </>
                        ) : (
                          <>
                            <ArrowDownCircle className="h-4 w-4 text-blue-500" />
                            <Badge variant="secondary">
                              Usage
                            </Badge>
                          </>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span
                        className={
                          transaction.transaction_type === 'purchase'
                            ? 'text-green-600 font-semibold'
                            : 'text-blue-600 font-semibold'
                        }
                      >
                        {transaction.transaction_type === 'purchase' ? '+' : '-'}
                        {transaction.tokens}
                      </span>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDate(transaction.created_at)}
                    </TableCell>
                    <TableCell>
                      {transaction.stripe_payment_intent_id && (
                        <code className="text-xs bg-muted px-2 py-1 rounded">
                          {transaction.stripe_payment_intent_id.substring(0, 20)}...
                        </code>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}