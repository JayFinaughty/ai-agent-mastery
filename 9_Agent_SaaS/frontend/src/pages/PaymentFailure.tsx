import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { XCircle } from 'lucide-react';

export default function PaymentFailure() {
  const navigate = useNavigate();
  const errorMessage = new URLSearchParams(window.location.search).get('error') ||
    'Your payment was not successful. Please try again.';

  return (
    <div className="container mx-auto p-6 max-w-2xl">
      <Card className="mt-12">
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            <XCircle className="h-16 w-16 text-red-500" />
          </div>
          <CardTitle className="text-center text-3xl">
            Payment Failed
          </CardTitle>
          <CardDescription className="text-center text-lg">
            We couldn't process your payment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertDescription className="text-center">
              {errorMessage}
            </AlertDescription>
          </Alert>
        </CardContent>
        <CardFooter className="flex justify-center gap-4">
          <Button onClick={() => navigate('/purchase-tokens')} size="lg">
            Try Again
          </Button>
          <Button onClick={() => navigate('/')} variant="outline" size="lg">
            Back to Chat
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}