import * as React from "react"
import { cn } from "../../lib/utils"

/**
 * Card - Design system compliant card component
 * 
 * Follows MedTrack design system:
 * - Background: white
 * - Border: neutral-200
 * - Border radius: rounded-2xl (16px)
 * - Shadow: shadow-soft
 * - Padding: p-6 lg:p-8 (responsive)
 * 
 * Note: For dashboard cards with animations, use DashboardCard component instead
 * 
 * @param {Object} props
 * @param {string} props.className - Additional classes
 * @returns {JSX.Element}
 */
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "bg-white border border-neutral-200 rounded-2xl shadow-soft",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

/**
 * CardHeader - Header section of card
 * Uses 8px grid spacing (space-y-2 = 8px)
 */
const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-2 p-6 lg:p-8", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

/**
 * CardTitle - Title text in card header
 * Typography: text-xl font-semibold (20px, 600 weight)
 */
const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-xl font-semibold text-neutral-900 tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

/**
 * CardDescription - Description text in card header
 * Typography: text-sm text-neutral-600 (14px, neutral-600)
 */
const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-neutral-600", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

/**
 * CardContent - Main content area of card
 * Padding: p-6 lg:p-8 (responsive, matches header)
 */
const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 lg:p-8 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

/**
 * CardFooter - Footer section of card
 * Padding: p-6 lg:p-8 (responsive, matches header)
 */
const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 lg:p-8 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }





