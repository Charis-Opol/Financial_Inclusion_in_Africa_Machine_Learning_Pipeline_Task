"""Post-deployment monitoring: reference profiles and PSI drift checks over the prediction log.

Like `fin_inclusion.serving`, this package needs only numpy (+ the serving
package), so the drift job runs from the same slim image as the API.
"""
