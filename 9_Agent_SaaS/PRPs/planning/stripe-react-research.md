# Stripe React Integration Research

This document provides comprehensive research on implementing Stripe payment flows in React applications using `@stripe/react-stripe-js` and `@stripe/stripe-js`.

## 1. Package Installation and Setup

### Required Packages

```bash
# Using npm
npm install --save @stripe/react-stripe-js @stripe/stripe-js

# Using yarn
yarn add @stripe/react-stripe-js @stripe/stripe-js
```

### Package Details
- **@stripe/react-stripe-js**: Latest version 4.0.2 (as of search date)
- **@stripe/stripe-js**: Core Stripe.js library with TypeScript support
- **Minimum React Version**: v16.8+ (hooks support required)
- **TypeScript Support**: Full TypeScript declarations included

### Basic Stripe Initialization

```tsx
import { loadStripe } from '@stripe/stripe-js';
import { Elements } from '@stripe/react-stripe-js';

// Initialize Stripe (do this outside component to avoid recreating)
const stripePromise = loadStripe('pk_test_your_publishable_key_here');

function App() {
  return (
    <Elements stripe={stripePromise}>
      <CheckoutForm />
    </Elements>
  );
}
```

### Provider Setup with Options

```tsx
import { Elements } from '@stripe/react-stripe-js';
import { loadStripe, StripeElementsOptions } from '@stripe/stripe-js';

const stripePromise = loadStripe('pk_test_your_publishable_key_here');

const options: StripeElementsOptions = {
  mode: 'payment',
  amount: 1099, // Amount in cents
  currency: 'usd',
  appearance: {
    theme: 'flat',
    variables: {
      colorPrimary: '#0570de',
      colorBackground: '#ffffff',
      colorText: '#30313d',
    },
  },
};

function App() {
  return (
    <Elements stripe={stripePromise} options={options}>
      <CheckoutForm />
    </Elements>
  );
}
```

**Documentation References:**
- Official React Stripe.js Reference: https://docs.stripe.com/sdks/stripejs-react
- NPM Package: https://www.npmjs.com/package/@stripe/react-stripe-js
- GitHub Repository: https://github.com/stripe/react-stripe-js

## 2. Payment Flow Implementation

### Creating Payment Intents from Frontend

**Server-Side Endpoint Example (Node.js):**
```javascript
// Server-side route to create PaymentIntent
app.post('/create-payment-intent', async (req, res) => {
  try {
    const { amount, currency = 'usd' } = req.body;

    const paymentIntent = await stripe.paymentIntents.create({
      amount: amount,
      currency: currency,
      metadata: { integration_check: 'accept_a_payment' },
    });

    res.json({
      clientSecret: paymentIntent.client_secret
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

### Using PaymentElement (Recommended)

```tsx
import React, { useState } from 'react';
import {
  PaymentElement,
  useStripe,
  useElements
} from '@stripe/react-stripe-js';

interface CheckoutFormProps {
  clientSecret: string;
}

const CheckoutForm: React.FC<CheckoutFormProps> = ({ clientSecret }) => {
  const stripe = useStripe();
  const elements = useElements();
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string>('');

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setIsLoading(true);
    setErrorMessage('');

    // Validate form before payment submission
    const { error: submitError } = await elements.submit();
    if (submitError) {
      setErrorMessage(submitError.message || 'An error occurred');
      setIsLoading(false);
      return;
    }

    // Confirm payment
    const { error } = await stripe.confirmPayment({
      elements,
      clientSecret,
      confirmParams: {
        return_url: `${window.location.origin}/payment-success`,
      },
    });

    if (error) {
      setErrorMessage(error.message || 'An unexpected error occurred');
    }

    setIsLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <PaymentElement />
      {errorMessage && (
        <div className="error-message" role="alert">
          {errorMessage}
        </div>
      )}
      <button
        type="submit"
        disabled={!stripe || !elements || isLoading}
        className="submit-button"
      >
        {isLoading ? 'Processing...' : 'Pay Now'}
      </button>
    </form>
  );
};
```

### Alternative: Using CardElement

```tsx
import { CardElement, useStripe, useElements } from '@stripe/react-stripe-js';

const CardForm: React.FC = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!stripe || !elements) return;

    setIsProcessing(true);

    const cardElement = elements.getElement(CardElement);

    if (!cardElement) {
      setIsProcessing(false);
      return;
    }

    // Create payment method
    const { error, paymentMethod } = await stripe.createPaymentMethod({
      type: 'card',
      card: cardElement,
    });

    if (error) {
      console.error('Error creating payment method:', error);
      setIsProcessing(false);
      return;
    }

    // Create PaymentIntent on server
    const response = await fetch('/create-payment-intent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount: 2000, // $20.00
        payment_method_id: paymentMethod.id,
      }),
    });

    const { client_secret } = await response.json();

    // Confirm payment
    const result = await stripe.confirmCardPayment(client_secret);

    setIsProcessing(false);

    if (result.error) {
      console.error('Payment failed:', result.error);
    } else {
      console.log('Payment succeeded:', result.paymentIntent);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <CardElement
        options={{
          style: {
            base: {
              fontSize: '16px',
              color: '#424770',
              '::placeholder': {
                color: '#aab7c4',
              },
            },
          },
        }}
      />
      <button type="submit" disabled={!stripe || isProcessing}>
        {isProcessing ? 'Processing...' : 'Pay'}
      </button>
    </form>
  );
};
```

### Redirect Flows vs Embedded Forms

**Redirect Flow (Checkout Sessions):**
```tsx
// Redirect to Stripe Checkout
const handleCheckout = async () => {
  const response = await fetch('/create-checkout-session', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      price_id: 'price_1234',
      quantity: 1,
    }),
  });

  const { url } = await response.json();
  window.location.href = url;
};
```

**Embedded Forms (Payment Intents):**
- More control over UI/UX
- Custom styling and validation
- Requires handling payment confirmation in your app
- Better for single-page applications

**Documentation References:**
- Payment Element Guide: https://docs.stripe.com/payments/payment-element
- Accept a Payment: https://docs.stripe.com/payments/accept-a-payment?platform=web&ui=elements

## 3. Success and Failure Handling

### Payment Success Callbacks

```tsx
import { useEffect } from 'react';
import { useStripe } from '@stripe/react-stripe-js';

const PaymentSuccess: React.FC = () => {
  const stripe = useStripe();

  useEffect(() => {
    if (!stripe) return;

    // Retrieve PaymentIntent from URL parameters
    const clientSecret = new URLSearchParams(window.location.search).get(
      'payment_intent_client_secret'
    );

    if (!clientSecret) return;

    stripe.retrievePaymentIntent(clientSecret).then(({ paymentIntent }) => {
      if (!paymentIntent) return;

      switch (paymentIntent.status) {
        case 'succeeded':
          setMessage('Payment succeeded!');
          // Update UI, send confirmation email, etc.
          break;
        case 'processing':
          setMessage('Your payment is processing.');
          break;
        case 'requires_payment_method':
          setMessage('Your payment was not successful, please try again.');
          break;
        default:
          setMessage('Something went wrong.');
          break;
      }
    });
  }, [stripe]);

  return <div>{message}</div>;
};
```

### Error Handling Patterns

```tsx
import { StripeError } from '@stripe/stripe-js';

const handleStripeError = (error: StripeError) => {
  switch (error.type) {
    case 'card_error':
      // Card was declined
      return `Payment failed: ${error.message}`;

    case 'validation_error':
      // Invalid parameters were supplied
      return `Validation error: ${error.message}`;

    case 'invalid_request_error':
      // Invalid request
      return 'Invalid request. Please try again.';

    case 'api_connection_error':
      // Network communication failed
      return 'Network error. Please check your connection.';

    case 'api_error':
      // Generic API error
      return 'An error occurred. Please try again.';

    case 'authentication_error':
      // Authentication failed
      return 'Authentication failed.';

    case 'rate_limit_error':
      // Too many requests
      return 'Too many requests. Please try again later.';

    default:
      return 'An unexpected error occurred.';
  }
};

// Usage in component
const [error, setError] = useState<string>('');

const processPayment = async () => {
  try {
    const result = await stripe.confirmPayment({
      elements,
      confirmParams: {
        return_url: 'https://example.com/success',
      },
    });

    if (result.error) {
      setError(handleStripeError(result.error));
    }
  } catch (err) {
    setError('An unexpected error occurred');
    console.error('Payment error:', err);
  }
};
```

### Redirect After Payment

```tsx
// Set up success and cancel URLs
const createCheckoutSession = async () => {
  const response = await fetch('/create-checkout-session', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      success_url: `${window.location.origin}/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${window.location.origin}/cancel`,
    }),
  });
};

// Handle success page
const SuccessPage: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>('');

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const session_id = urlParams.get('session_id');

    if (session_id) {
      setSessionId(session_id);
      // Retrieve session details from your server
      fetchSessionDetails(session_id);
    }
  }, []);

  return (
    <div>
      <h1>Payment Successful!</h1>
      {sessionId && <p>Session ID: {sessionId}</p>}
    </div>
  );
};
```

**Documentation References:**
- Error Handling: https://docs.stripe.com/error-handling

## 4. TypeScript Integration

### Type Definitions

```tsx
import {
  Stripe,
  StripeElements,
  PaymentIntent,
  SetupIntent,
  PaymentMethod,
  StripeError,
  StripeCardElement,
  StripeCardElementChangeEvent,
} from '@stripe/stripe-js';

// Component props with types
interface CheckoutFormProps {
  clientSecret: string;
  amount: number;
  currency: string;
  onPaymentSuccess: (paymentIntent: PaymentIntent) => void;
  onPaymentError: (error: StripeError) => void;
}

// Custom hook with types
const usePaymentProcessing = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<StripeError | null>(null);

  const processPayment = async (
    clientSecret: string
  ): Promise<{ paymentIntent?: PaymentIntent; error?: StripeError }> => {
    if (!stripe || !elements) {
      throw new Error('Stripe not initialized');
    }

    setIsProcessing(true);
    setError(null);

    try {
      const result = await stripe.confirmPayment({
        elements,
        clientSecret,
        redirect: 'if_required',
      });

      setIsProcessing(false);

      if (result.error) {
        setError(result.error);
        return { error: result.error };
      }

      return { paymentIntent: result.paymentIntent };
    } catch (err) {
      setIsProcessing(false);
      const error = err as StripeError;
      setError(error);
      return { error };
    }
  };

  return {
    processPayment,
    isProcessing,
    error,
  };
};
```

### Type-Safe Payment Handling

```tsx
import { PaymentIntentResult, SetupIntentResult } from '@stripe/stripe-js';

// Type-safe payment confirmation
const confirmPaymentWithTypes = async (
  stripe: Stripe,
  elements: StripeElements,
  clientSecret: string
): Promise<PaymentIntentResult> => {
  return await stripe.confirmPayment({
    elements,
    clientSecret,
    confirmParams: {
      return_url: window.location.origin + '/success',
    },
  });
};

// Type-safe event handlers
const handleCardElementChange = (event: StripeCardElementChangeEvent) => {
  if (event.error) {
    setError(event.error.message);
  } else {
    setError('');
  }

  setIsCardComplete(event.complete);
  setCardBrand(event.brand);
};

// Type-safe form data
interface PaymentFormData {
  amount: number;
  currency: 'usd' | 'eur' | 'gbp';
  payment_method_types: Array<'card' | 'ideal' | 'sepa_debit'>;
  metadata?: Record<string, string>;
}

const createPaymentIntent = async (data: PaymentFormData) => {
  const response = await fetch('/create-payment-intent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('Failed to create payment intent');
  }

  return await response.json();
};
```

### Advanced TypeScript Examples

```tsx
// Generic payment component
interface PaymentComponentProps<T = any> {
  onSuccess: (result: T) => void;
  onError: (error: StripeError) => void;
  metadata?: Record<string, any>;
}

const PaymentComponent = <T extends PaymentIntent | SetupIntent>({
  onSuccess,
  onError,
  metadata,
}: PaymentComponentProps<T>) => {
  // Component implementation
};

// Typed Stripe context
interface StripeContextValue {
  stripe: Stripe | null;
  elements: StripeElements | null;
  isLoaded: boolean;
  error: StripeError | null;
}

const StripeContext = React.createContext<StripeContextValue | null>(null);

// Custom hook with proper typing
export const useStripeContext = (): StripeContextValue => {
  const context = useContext(StripeContext);
  if (!context) {
    throw new Error('useStripeContext must be used within StripeProvider');
  }
  return context;
};
```

**Key Type Exports:**
- `Stripe` - Main Stripe instance
- `StripeElements` - Elements instance
- `PaymentIntent`, `SetupIntent` - Intent objects
- `PaymentMethod` - Payment method objects
- `StripeError` - Error objects
- Element-specific types: `StripeCardElement`, `StripePaymentElement`, etc.

## 5. Best Practices

### Security Considerations

**Publishable Keys:**
```tsx
// ✅ Correct: Use environment variables
const STRIPE_PUBLISHABLE_KEY = process.env.REACT_APP_STRIPE_PUBLISHABLE_KEY;

// ✅ Safe to use on client-side
const stripePromise = loadStripe(STRIPE_PUBLISHABLE_KEY!);

// ❌ Never do this: Hardcode keys
const stripePromise = loadStripe('pk_test_12345...'); // Don't hardcode

// ❌ Never expose secret keys on frontend
const SECRET_KEY = 'sk_test_...'; // This should NEVER be in frontend code
```

**Environment Variable Setup (Create React App):**
```bash
# .env.local
REACT_APP_STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
```

**Key Management Best Practices:**
- Store API keys in environment variables
- Never commit keys to version control
- Use different keys for development/production
- Rotate keys periodically
- Use Key Management Systems (KMS) for production

### Loading States and UX Patterns

```tsx
const PaymentForm: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [isStripeLoading, setIsStripeLoading] = useState(true);
  const stripe = useStripe();
  const elements = useElements();

  // Handle Stripe loading state
  useEffect(() => {
    if (stripe && elements) {
      setIsStripeLoading(false);
    }
  }, [stripe, elements]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);

    try {
      // Payment processing logic
      await processPayment();
    } finally {
      setIsLoading(false);
    }
  };

  // Show loading spinner while Stripe loads
  if (isStripeLoading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <p>Loading payment form...</p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit}>
      <PaymentElement />

      <button
        type="submit"
        disabled={!stripe || !elements || isLoading}
        className={`pay-button ${isLoading ? 'loading' : ''}`}
      >
        {isLoading ? (
          <>
            <div className="button-spinner" />
            Processing...
          </>
        ) : (
          'Pay Now'
        )}
      </button>
    </form>
  );
};
```

### Error Messaging

```tsx
// User-friendly error messages
const getErrorMessage = (error: StripeError): string => {
  const errorMessages: Record<string, string> = {
    'card_declined': 'Your card was declined. Please try a different card.',
    'expired_card': 'Your card has expired. Please use a different card.',
    'insufficient_funds': 'Your card has insufficient funds.',
    'invalid_cvc': 'Your card\'s security code is invalid.',
    'invalid_expiry_month': 'Your card\'s expiration month is invalid.',
    'invalid_expiry_year': 'Your card\'s expiration year is invalid.',
    'invalid_number': 'Your card number is invalid.',
    'processing_error': 'An error occurred while processing your card.',
  };

  return errorMessages[error.code] || 'An unexpected error occurred. Please try again.';
};

// Error display component
const ErrorMessage: React.FC<{ error: string }> = ({ error }) => (
  <div
    className="error-message"
    role="alert"
    style={{
      color: '#e74c3c',
      backgroundColor: '#fdf2f2',
      border: '1px solid #fecaca',
      borderRadius: '4px',
      padding: '12px',
      marginTop: '8px',
    }}
  >
    {error}
  </div>
);
```

### Mobile Responsiveness

```css
/* Responsive Stripe Elements styling */
.payment-form {
  max-width: 400px;
  margin: 0 auto;
  padding: 20px;
}

.StripeElement {
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  margin: 10px 0;
  transition: border-color 0.3s;
}

.StripeElement:focus {
  border-color: #0570de;
  outline: none;
  box-shadow: 0 0 0 2px rgba(5, 112, 222, 0.1);
}

.StripeElement--invalid {
  border-color: #e74c3c;
}

/* Mobile-specific styles */
@media (max-width: 768px) {
  .payment-form {
    padding: 16px;
    margin: 0 16px;
  }

  .pay-button {
    width: 100%;
    padding: 14px;
    font-size: 16px; /* Prevents zoom on iOS */
  }
}

/* Touch-friendly button sizing */
.pay-button {
  min-height: 44px; /* iOS minimum touch target */
  min-width: 44px;
  font-size: 16px;
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  background: #0570de;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s;
}

.pay-button:hover:not(:disabled) {
  background: #045bb5;
}

.pay-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
```

### Performance Optimization

```tsx
// Memoize Stripe promise to prevent recreation
const stripePromise = useMemo(() => {
  return loadStripe(process.env.REACT_APP_STRIPE_PUBLISHABLE_KEY!);
}, []);

// Lazy load payment form
const PaymentForm = React.lazy(() => import('./PaymentForm'));

const CheckoutPage: React.FC = () => {
  return (
    <Elements stripe={stripePromise}>
      <Suspense fallback={<div>Loading payment form...</div>}>
        <PaymentForm />
      </Suspense>
    </Elements>
  );
};

// Debounce validation for better UX
import { useDebounce } from './hooks/useDebounce';

const useCardValidation = () => {
  const [cardState, setCardState] = useState({
    error: null,
    complete: false,
  });

  const debouncedValidation = useDebounce((event) => {
    // Validation logic here
  }, 300);

  return { cardState, debouncedValidation };
};
```

### Testing Best Practices

```tsx
// Test card numbers for different scenarios
const TEST_CARDS = {
  VISA_SUCCESS: '4242424242424242',
  VISA_DECLINED: '4000000000000002',
  INSUFFICIENT_FUNDS: '4000000000009995',
  EXPIRED_CARD: '4000000000000069',
  INCORRECT_CVC: '4000000000000127',
};

// Mock Stripe for testing
jest.mock('@stripe/stripe-js', () => ({
  loadStripe: jest.fn(() =>
    Promise.resolve({
      confirmPayment: jest.fn(),
      createPaymentMethod: jest.fn(),
    })
  ),
}));

// Test component
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Elements } from '@stripe/react-stripe-js';

const mockStripe = {
  confirmPayment: jest.fn(),
  elements: jest.fn(),
};

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Elements stripe={mockStripe as any}>
    {children}
  </Elements>
);

test('handles payment submission', async () => {
  mockStripe.confirmPayment.mockResolvedValue({
    paymentIntent: { status: 'succeeded' },
  });

  render(
    <TestWrapper>
      <PaymentForm clientSecret="test_client_secret" />
    </TestWrapper>
  );

  fireEvent.click(screen.getByText('Pay Now'));

  await waitFor(() => {
    expect(mockStripe.confirmPayment).toHaveBeenCalled();
  });
});
```

## Common Pitfalls to Avoid

### 1. Stripe Promise Recreation
```tsx
// ❌ Don't recreate Stripe promise on every render
const App = () => {
  const stripePromise = loadStripe('pk_test_...'); // Creates new promise each render

  return <Elements stripe={stripePromise}>...</Elements>;
};

// ✅ Create promise outside component or use useMemo
const stripePromise = loadStripe('pk_test_...');
const App = () => <Elements stripe={stripePromise}>...</Elements>;
```

### 2. Missing Error Handling
```tsx
// ❌ Not handling Stripe loading state
const CheckoutForm = () => {
  const stripe = useStripe();
  const elements = useElements();

  const handleSubmit = async () => {
    // This will fail if stripe/elements aren't loaded yet
    await stripe.confirmPayment(...);
  };
};

// ✅ Proper loading state handling
const CheckoutForm = () => {
  const stripe = useStripe();
  const elements = useElements();

  const handleSubmit = async () => {
    if (!stripe || !elements) {
      return; // Stripe hasn't loaded yet
    }

    await stripe.confirmPayment(...);
  };
};
```

### 3. Improper Element Mounting
```tsx
// ❌ Mounting elements outside Elements provider
const App = () => (
  <div>
    <PaymentElement /> {/* This won't work */}
    <Elements stripe={stripePromise}>
      <CheckoutForm />
    </Elements>
  </div>
);

// ✅ Mount elements inside provider
const App = () => (
  <Elements stripe={stripePromise}>
    <PaymentElement />
    <CheckoutForm />
  </Elements>
);
```

## Key Resources and References

### Official Documentation
- **React Stripe.js Reference**: https://docs.stripe.com/sdks/stripejs-react
- **Payment Element Guide**: https://docs.stripe.com/payments/payment-element
- **Accept a Payment Guide**: https://docs.stripe.com/payments/accept-a-payment?platform=web&ui=elements
- **Error Handling**: https://docs.stripe.com/error-handling
- **API Keys Best Practices**: https://docs.stripe.com/keys-best-practices

### Package Resources
- **NPM Package**: https://www.npmjs.com/package/@stripe/react-stripe-js
- **GitHub Repository**: https://github.com/stripe/react-stripe-js
- **Stripe.js SDK**: https://www.npmjs.com/package/@stripe/stripe-js

### Additional Learning Resources
- **Stripe JS Reference**: https://docs.stripe.com/js
- **Stripe Elements Customization**: https://docs.stripe.com/stripe-js/appearance-api
- **Stripe Webhooks**: https://docs.stripe.com/webhooks
- **Testing with Stripe**: https://docs.stripe.com/testing

---

*This research document provides a comprehensive foundation for implementing Stripe payment flows in React applications. Always refer to the latest Stripe documentation for the most up-to-date information and best practices.*