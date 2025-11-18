import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

/**
 * Badge - Design system compliant badge component
 * 
 * Follows MedTrack design system:
 * - Uses design token colors (primary, medical, error, warning, neutral)
 * - Border radius: rounded-full for pills, rounded-lg for squares
 * - Typography: text-xs font-semibold
 * - Padding: px-2.5 py-0.5 (on 8px grid)
 * 
 * @param {Object} props
 * @param {string} props.variant - Badge variant
 * @param {string} props.className - Additional classes
 * @returns {JSX.Element}
 */
const badgeVariants = cva(
  "inline-flex items-center rounded-full border-0 px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default:
          "bg-primary-50 text-primary-700",
        primary:
          "bg-primary-600 text-white",
        secondary:
          "bg-neutral-100 text-neutral-700",
        success:
          "bg-medical-50 text-medical-700",
        medical:
          "bg-medical-50 text-medical-700",
        error:
          "bg-error-50 text-error-700",
        warning:
          "bg-warning-50 text-warning-700",
        info:
          "bg-primary-50 text-primary-700",
        outline:
          "bg-transparent border border-neutral-200 text-neutral-700",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({ className, variant, ...props }) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }





