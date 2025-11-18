import * as React from "react"
import { cn } from "../../lib/utils"

/**
 * Input - Design system compliant input component
 * 
 * Follows MedTrack design system:
 * - Height: 44px (h-11) for proper touch targets
 * - Border: neutral-200
 * - Focus: primary-500 ring
 * - Typography: text-base (16px minimum)
 * - Border radius: rounded-md (8px)
 * 
 * @param {Object} props
 * @param {string} props.className - Additional classes
 * @param {string} props.type - Input type
 * @param {boolean} props.error - Error state
 * @param {string} props.helper - Helper text
 * @returns {JSX.Element}
 */
const Input = React.forwardRef(({ className, type, error, helper, ...props }, ref) => {
  return (
    <div className="w-full">
      <input
        type={type}
        className={cn(
          // Base styles - Design system compliant
          "flex h-11 w-full rounded-md border bg-white px-4 text-base",
          "placeholder:text-neutral-400",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 focus-visible:border-transparent",
          "disabled:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-50",
          "transition-colors duration-200",
          // Error state
          error 
            ? "border-error-500 focus-visible:ring-error-500" 
            : "border-neutral-200",
          className
        )}
        ref={ref}
        aria-invalid={error ? "true" : "false"}
        {...props}
      />
      {helper && !error && (
        <p className="mt-2 text-sm text-neutral-500">{helper}</p>
      )}
    </div>
  )
})
Input.displayName = "Input"

export { Input }





