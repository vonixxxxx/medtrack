import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"
import { Loader2 } from "lucide-react"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        primary:
          "bg-primary-600 text-white shadow-lg shadow-primary-600/25 hover:bg-primary-700 hover:shadow-xl hover:shadow-primary-600/30 hover:-translate-y-0.5 active:translate-y-0 active:shadow-lg",
        secondary:
          "bg-white text-primary-700 border-2 border-primary-200 hover:bg-primary-50 hover:border-primary-300 active:bg-primary-100",
        tertiary:
          "bg-transparent text-primary-700 hover:bg-primary-50 active:bg-primary-100",
        destructive:
          "bg-error-500 text-white shadow-lg shadow-error-500/25 hover:bg-error-600 hover:shadow-xl active:translate-y-0",
        success:
          "bg-medical-600 text-white shadow-lg shadow-medical-600/25 hover:bg-medical-700 hover:shadow-xl active:translate-y-0",
        ghost:
          "hover:bg-neutral-100 text-neutral-700 hover:text-neutral-900",
        default: "bg-primary-600 text-white hover:bg-primary-700",
        outline:
          "border border-neutral-200 bg-white hover:bg-neutral-50",
        link: "text-primary-600 underline-offset-4 hover:underline",
      },
      size: {
        sm: "h-9 px-4 text-xs",
        md: "h-11 px-6 text-sm",
        lg: "h-12 px-8 text-base",
        icon: "h-11 w-11",
        default: "h-10 px-4 py-2",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
)

const Button = React.forwardRef(({ className, variant, size, asChild = false, loading, children, disabled, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"
  return (
    <Comp
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          {children}
        </>
      ) : (
        children
      )}
    </Comp>
  )
})
Button.displayName = "Button"

export { Button, buttonVariants }





