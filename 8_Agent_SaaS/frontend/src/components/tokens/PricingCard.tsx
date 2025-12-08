import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface PricingCardProps {
  tier: string;
  name: string;
  price: number;
  tokens: number;
  selected: boolean;
  onClick: () => void;
  loading?: boolean;
}

export default function PricingCard({
  tier,
  name,
  price,
  tokens,
  selected,
  onClick,
  loading = false
}: PricingCardProps) {
  const tokensPerDollar = Math.round(tokens / price);

  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-lg ${
        selected ? 'ring-2 ring-primary' : ''
      }`}
      onClick={onClick}
    >
      <CardHeader>
        <div className="flex justify-between items-start">
          <CardTitle>{name}</CardTitle>
          {selected && (
            <Badge variant="default">Selected</Badge>
          )}
        </div>
        <CardDescription>
          {tokensPerDollar} tokens per dollar
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="text-4xl font-bold">
            ${price}
          </div>
          <div className="text-xl text-muted-foreground">
            {tokens} tokens
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button
          className="w-full"
          variant={selected ? "default" : "outline"}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Select'}
        </Button>
      </CardFooter>
    </Card>
  );
}